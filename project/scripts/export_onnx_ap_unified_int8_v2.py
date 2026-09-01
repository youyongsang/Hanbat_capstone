"""Build a single-graph (If-node) Early Exit ONNX model with genuinely
INT8-quantized LSTM layers.

Background: export_onnx_ap_unified.py produces one ONNX graph with `If`
nodes (see docs/yongsang/onnx_early_exit_redesign.{md,html}) that beats the staged
(3-session) export on Pi latency. Naively INT8-quantizing *that* graph
(export_onnx_ap_unified_int8.py) leaves accuracy unchanged but gives no
speedup: onnxruntime's dynamic quantizer silently fails to convert LSTM to
`DynamicQuantizeLSTM` once the graph also contains control-flow (`If`)
nodes -- confirmed by inspecting the quantized graph's ops recursively (only
the tiny classifier1 Gemm got quantized; all three LSTM nodes stayed float).

This script gets both wins at once: it quantizes each of the three *staged*
(flat, no control flow) stage graphs independently -- where the quantizer
does correctly emit `DynamicQuantizeLSTM` -- then hand-assembles them back
into one If-node graph using onnx.helper, re-deriving the entropy/threshold/
If wiring that torch.jit.script generated for the fp32 unified export
(Softmax -> +eps -> Log -> Mul -> ReduceSum -> Neg -> Gather(0) -> Less(theta)
-> Cast -> If). Each stage's internal tensor names are prefixed to avoid
collisions; the boundary tensors ("input"/"hidden1"/"hidden2"/"exit1"/
"exit2"/"exit3") are left as-is since the stage exports already share them
by convention.

Prerequisite: run `python project/scripts/export_onnx_ap.py` first (produces
the staged fp32 `ap_early_exit_{fixed,dynamic}_stage{1,2,3}.onnx` files this
script quantizes and reassembles). This script quantizes those stages itself.

Verified (2026-08-28): PyTorch reference match on the full ap_metrics_v2_redesign2
test set -- 309/310 exact (1 borderline-entropy mismatch from int8 noise),
accuracy identical to fp32 (88.4%/89.0%). Pi latency: ~0.64ms (fixed) /
~0.68ms (dynamic) vs. 1.18ms/1.19ms for the fp32 unified graph and 1.97ms
baseline -- roughly another 45% off the fp32 unified graph, ~65% off baseline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "ap_v2_redesign2"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.ap_features import AP_FEATURE_COLUMNS, WINDOW_SIZE  # noqa: E402

INPUT_SIZE = len(AP_FEATURE_COLUMNS)

THETA_1 = 0.3
THETA_2 = 0.6


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def entropy_and_cond_nodes(exit_tensor: str, theta, prefix: str) -> tuple[list, str]:
    """Softmax->+eps->log->mul->reducesum->neg->gather0->less(theta)->cast(bool).
    `theta` is either a float constant or the name of an existing tensor
    (e.g. a dynamically computed threshold). Returns (nodes, cond_tensor_name)."""
    n = []
    softmax = f"{prefix}_softmax"
    eps_c = f"{prefix}_eps"
    added = f"{prefix}_added"
    logged = f"{prefix}_logged"
    muled = f"{prefix}_muled"
    axis_c = f"{prefix}_axis"
    summed = f"{prefix}_summed"
    negged = f"{prefix}_negged"
    idx0_c = f"{prefix}_idx0"
    gathered = f"{prefix}_entropy0"
    less_out = f"{prefix}_less"
    cond = f"{prefix}_cond"

    n.append(helper.make_node("Softmax", [exit_tensor], [softmax], axis=-1))
    n.append(helper.make_node("Constant", [], [eps_c], value=helper.make_tensor(eps_c, TensorProto.FLOAT, [], [1e-8])))
    n.append(helper.make_node("Add", [softmax, eps_c], [added]))
    n.append(helper.make_node("Log", [added], [logged]))
    n.append(helper.make_node("Mul", [softmax, logged], [muled]))
    n.append(helper.make_node("Constant", [], [axis_c], value=helper.make_tensor(axis_c, TensorProto.INT64, [1], [-1])))
    n.append(helper.make_node("ReduceSum", [muled, axis_c], [summed]))
    n.append(helper.make_node("Neg", [summed], [negged]))
    n.append(helper.make_node("Constant", [], [idx0_c], value=helper.make_tensor(idx0_c, TensorProto.INT64, [1], [0])))
    n.append(helper.make_node("Gather", [negged, idx0_c], [gathered], axis=0))
    if isinstance(theta, str):
        theta_ref = theta
    else:
        theta_c = f"{prefix}_theta"
        n.append(helper.make_node("Constant", [], [theta_c], value=helper.make_tensor(theta_c, TensorProto.FLOAT, [], [theta])))
        theta_ref = theta_c
    n.append(helper.make_node("Less", [gathered, theta_ref], [less_out]))
    n.append(helper.make_node("Cast", [less_out], [cond], to=TensorProto.BOOL))
    return n, cond


DYNAMIC_MIN_THRESHOLD = 0.22
DYNAMIC_RECENT_STEPS = 5
DYNAMIC_SPIKE_THRESHOLD = 0.25


def dynamic_theta_nodes(base_theta_1: float, base_theta_2: float) -> tuple[list, str, str]:
    """Mirrors compute_dynamic_threshold: adjust (theta_1, theta_2) from the
    input window's recent occupancy delta (feature index 1). Returns
    (nodes, theta_1_tensor_name, theta_2_tensor_name)."""
    n = []
    # occupancy = x[0, -RECENT_STEPS:, 1]  -> just need last two steps' delta
    # Slice input[0, -2:, 1:2] -> shape (2,1) -> Squeeze -> (2,)
    starts_c, ends_c, axes_c, steps_c = "dt_starts", "dt_ends", "dt_axes", "dt_steps"
    n.append(helper.make_node("Constant", [], [starts_c], value=helper.make_tensor(starts_c, TensorProto.INT64, [1], [-2])))
    n.append(helper.make_node("Constant", [], [ends_c], value=helper.make_tensor(ends_c, TensorProto.INT64, [1], [9223372036854775807])))
    n.append(helper.make_node("Constant", [], [axes_c], value=helper.make_tensor(axes_c, TensorProto.INT64, [1], [1])))
    n.append(helper.make_node("Constant", [], [steps_c], value=helper.make_tensor(steps_c, TensorProto.INT64, [1], [1])))
    sliced = "dt_occ_slice"  # (1, 2, 6)
    n.append(helper.make_node("Slice", ["input", starts_c, ends_c, axes_c, steps_c], [sliced]))

    idx1_c = "dt_idx1"
    n.append(helper.make_node("Constant", [], [idx1_c], value=helper.make_tensor(idx1_c, TensorProto.INT64, [], [1])))
    occ_col = "dt_occ_col"  # (1, 2)
    n.append(helper.make_node("Gather", [sliced, idx1_c], [occ_col], axis=2))

    idx0_c = "dt_idx0b"
    n.append(helper.make_node("Constant", [], [idx0_c], value=helper.make_tensor(idx0_c, TensorProto.INT64, [], [0])))
    occ_row = "dt_occ_row"  # (2,)
    n.append(helper.make_node("Gather", [occ_col, idx0_c], [occ_row], axis=0))

    last_idx_c, prev_idx_c = "dt_last_idx", "dt_prev_idx"
    n.append(helper.make_node("Constant", [], [last_idx_c], value=helper.make_tensor(last_idx_c, TensorProto.INT64, [], [1])))
    n.append(helper.make_node("Constant", [], [prev_idx_c], value=helper.make_tensor(prev_idx_c, TensorProto.INT64, [], [0])))
    last_v = "dt_last_v"
    prev_v = "dt_prev_v"
    n.append(helper.make_node("Gather", [occ_row, last_idx_c], [last_v], axis=0))
    n.append(helper.make_node("Gather", [occ_row, prev_idx_c], [prev_v], axis=0))

    diff = "dt_diff"
    absdiff = "dt_absdiff"
    n.append(helper.make_node("Sub", [last_v, prev_v], [diff]))
    n.append(helper.make_node("Abs", [diff], [absdiff]))

    spike_c = "dt_spike_thr"
    n.append(helper.make_node("Constant", [], [spike_c], value=helper.make_tensor(spike_c, TensorProto.FLOAT, [], [DYNAMIC_SPIKE_THRESHOLD])))
    is_spike = "dt_is_spike"
    n.append(helper.make_node("Greater", [absdiff, spike_c], [is_spike]))

    base1_c, base2_c = "dt_base1", "dt_base2"
    relaxed1_c, relaxed2_c = "dt_relaxed1", "dt_relaxed2"
    n.append(helper.make_node("Constant", [], [base1_c], value=helper.make_tensor(base1_c, TensorProto.FLOAT, [], [base_theta_1])))
    n.append(helper.make_node("Constant", [], [base2_c], value=helper.make_tensor(base2_c, TensorProto.FLOAT, [], [max(base_theta_1 * 1.25, DYNAMIC_MIN_THRESHOLD)])))
    n.append(helper.make_node("Constant", [], [relaxed1_c], value=helper.make_tensor(relaxed1_c, TensorProto.FLOAT, [], [base_theta_2])))
    n.append(helper.make_node("Constant", [], [relaxed2_c], value=helper.make_tensor(relaxed2_c, TensorProto.FLOAT, [], [max(base_theta_2 * 1.25, DYNAMIC_MIN_THRESHOLD * 2)])))

    theta1_out, theta2_out = "dt_theta1", "dt_theta2"
    # is_spike True -> base theta; False -> relaxed theta
    n.append(helper.make_node("Where", [is_spike, base1_c, base2_c], [theta1_out]))
    n.append(helper.make_node("Where", [is_spike, relaxed1_c, relaxed2_c], [theta2_out]))
    return n, theta1_out, theta2_out


def exit_point_const(value: int, name: str):
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name, TensorProto.INT64, [], [value]))


def quantize_stages(checkpoint_dir: Path, suffix: str) -> None:
    """INT8-quantize each flat staged export. Real LSTM quantization only
    works because these graphs have no control-flow (`If`) nodes."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for stage in (1, 2, 3):
        src = checkpoint_dir / f"ap_early_exit_{suffix}_stage{stage}.onnx"
        dst = checkpoint_dir / f"ap_early_exit_{suffix}_stage{stage}_int8.onnx"
        if not src.exists():
            raise FileNotFoundError(
                f"Not found: {display_path(src)} -- run export_onnx_ap.py first"
            )
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm", "LSTM"],
        )
        print(f"Stage quantized: {display_path(dst)}")


def load_stage(path: Path, prefix: str, keep: set[str]):
    """Load a stage graph's nodes+initializers, renaming every internal
    tensor with `prefix` so sibling stages never collide, except names in
    `keep` (the boundary I/O tensors that must stay as-is)."""
    m = onnx.load(str(path))
    nodes = list(m.graph.node)
    inits = list(m.graph.initializer)

    def ren(name: str) -> str:
        if name == "" or name in keep:
            return name
        return f"{prefix}{name}"

    new_nodes = []
    for n in nodes:
        nn = onnx.NodeProto()
        nn.CopyFrom(n)
        nn.input[:] = [ren(x) for x in n.input]
        nn.output[:] = [ren(x) for x in n.output]
        if nn.name:
            nn.name = f"{prefix}{nn.name}"
        new_nodes.append(nn)

    new_inits = []
    for it in inits:
        ni = onnx.TensorProto()
        ni.CopyFrom(it)
        ni.name = ren(it.name)
        new_inits.append(ni)

    return new_nodes, new_inits


def build(checkpoint_dir: Path, fixed: bool, out_path: Path) -> None:
    suffix = "fixed" if fixed else "dynamic"
    stage1_nodes, stage1_init = load_stage(checkpoint_dir / f"ap_early_exit_{suffix}_stage1_int8.onnx", "s1_", {"input", "hidden1", "exit1"})
    stage2_nodes, stage2_init = load_stage(checkpoint_dir / f"ap_early_exit_{suffix}_stage2_int8.onnx", "s2_", {"hidden1", "hidden2", "exit2"})
    stage3_nodes, stage3_init = load_stage(checkpoint_dir / f"ap_early_exit_{suffix}_stage3_int8.onnx", "s3_", {"hidden2", "exit3"})

    if fixed:
        dyn_nodes: list = []
        theta1_ref: object = THETA_1
        theta2_ref: object = THETA_2
    else:
        dyn_nodes, theta1_ref, theta2_ref = dynamic_theta_nodes(THETA_1, THETA_2)

    # ---- depth-2 (innermost) subgraph: stage3 + exit=3 ----
    exit3_const = exit_point_const(3, "exit3_point")
    depth2_nodes = list(stage3_nodes) + [exit3_const]
    depth2_graph = helper.make_graph(
        depth2_nodes, "depth2", [], [
            helper.make_tensor_value_info("exit3", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("exit3_point", TensorProto.INT64, []),
        ],
        initializer=stage3_init,
    )

    # ---- depth-1 else-branch subgraph: stage2 + entropy2/cond2 + If(depth2) ----
    ent2_nodes, cond2 = entropy_and_cond_nodes("exit2", theta2_ref, "s2")
    exit2_const = exit_point_const(2, "exit2_point")
    then2_graph = helper.make_graph(
        [
            helper.make_node("Identity", ["exit2"], ["then2_logits"]),
            helper.make_node("Identity", ["exit2_point"], ["then2_exit"]),
        ], "then2", [], [
            helper.make_tensor_value_info("then2_logits", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("then2_exit", TensorProto.INT64, []),
        ],
    )
    if2_node = helper.make_node(
        "If", [cond2], ["out_logits_2", "out_exit_2"],
        then_branch=then2_graph, else_branch=depth2_graph,
    )
    depth1_else_nodes = list(stage2_nodes) + ent2_nodes + [exit2_const, if2_node]
    depth1_else_graph = helper.make_graph(
        depth1_else_nodes, "depth1_else", [], [
            helper.make_tensor_value_info("out_logits_2", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("out_exit_2", TensorProto.INT64, []),
        ],
        initializer=stage2_init,
    )

    # ---- depth-0 (main) graph: stage1 + entropy1/cond1 + If(depth1) ----
    ent1_nodes, cond1 = entropy_and_cond_nodes("exit1", theta1_ref, "s1")
    exit1_const = exit_point_const(1, "exit1_point")
    then1_graph = helper.make_graph(
        [
            helper.make_node("Identity", ["exit1"], ["then1_logits"]),
            helper.make_node("Identity", ["exit1_point"], ["then1_exit"]),
        ], "then1", [], [
            helper.make_tensor_value_info("then1_logits", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("then1_exit", TensorProto.INT64, []),
        ],
    )
    if1_node = helper.make_node(
        "If", [cond1], ["logits", "exit_point"],
        then_branch=then1_graph, else_branch=depth1_else_graph,
    )
    main_nodes = dyn_nodes + list(stage1_nodes) + ent1_nodes + [exit1_const, if1_node]
    main_graph = helper.make_graph(
        main_nodes, "unified_int8",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, WINDOW_SIZE, INPUT_SIZE])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("exit_point", TensorProto.INT64, []),
        ],
        initializer=stage1_init,
    )

    model = helper.make_model(main_graph, opset_imports=[
        helper.make_opsetid("", 16),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"Built: {display_path(out_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build INT8 unified (If-node, quantized LSTM) AP Early Exit ONNX graphs."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)
    parser.add_argument(
        "--skip-stage-quantize",
        action="store_true",
        help="Skip re-quantizing the staged exports (reuse existing *_stage{1,2,3}_int8.onnx).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_stage_quantize:
        quantize_stages(args.checkpoint_dir, "fixed")
        quantize_stages(args.checkpoint_dir, "dynamic")
    build(args.checkpoint_dir, True, args.checkpoint_dir / "ap_early_exit_fixed_unified_int8_v2.onnx")
    build(args.checkpoint_dir, False, args.checkpoint_dir / "ap_early_exit_dynamic_unified_int8_v2.onnx")


if __name__ == "__main__":
    main()
