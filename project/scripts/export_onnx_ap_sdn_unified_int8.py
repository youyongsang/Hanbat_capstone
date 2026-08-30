"""Build a single-graph (If-node), INT8-quantized SDN-style ONNX model.

Same fix as export_onnx_ap_unified_int8_v2.py for the Early Exit model:
quantize each flat staged SDN graph independently (where the ORT quantizer
correctly emits DynamicQuantizeLSTM), then hand-assemble the quantized
pieces into one If-node graph. SDN's exit rule is confidence-based (exit
when max softmax prob >= threshold), the mirror image of Early Exit's
entropy-based rule (exit when entropy < threshold) -- so the branch wiring
here is Softmax -> ReduceMax -> GreaterOrEqual(threshold) instead of
Softmax -> entropy -> Less(theta).

Prerequisite: run `python project/scripts/export_onnx_ap_sdn.py` first.
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
from utils.ap_features import AP_FEATURE_COLUMNS  # noqa: E402

INPUT_SIZE = len(AP_FEATURE_COLUMNS)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def quantize_stages(checkpoint_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for stage in (1, 2, 3):
        src = checkpoint_dir / f"ap_sdn_stage{stage}.onnx"
        dst = checkpoint_dir / f"ap_sdn_stage{stage}_int8.onnx"
        if not src.exists():
            raise FileNotFoundError(f"Not found: {display_path(src)} -- run export_onnx_ap_sdn.py first")
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm", "LSTM"],
        )
        print(f"Stage quantized: {display_path(dst)}")


def confidence_and_cond_nodes(exit_tensor: str, threshold: float, prefix: str) -> tuple[list, str]:
    """Softmax -> ReduceMax(confidence) -> GreaterOrEqual(threshold) -> cond."""
    n = []
    softmax = f"{prefix}_softmax"
    conf = f"{prefix}_conf"
    thr_c = f"{prefix}_thr"
    cond = f"{prefix}_cond"

    n.append(helper.make_node("Softmax", [exit_tensor], [softmax], axis=-1))
    n.append(helper.make_node("ReduceMax", [softmax], [conf], axes=[-1], keepdims=0))
    n.append(helper.make_node("Constant", [], [thr_c], value=helper.make_tensor(thr_c, TensorProto.FLOAT, [], [threshold])))
    n.append(helper.make_node("GreaterOrEqual", [conf, thr_c], [cond]))
    return n, cond


def exit_point_const(value: int, name: str):
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name, TensorProto.INT64, [], [value]))


def load_stage(path: Path, prefix: str, keep: set[str]):
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


def build(checkpoint_dir: Path, threshold: float, out_path: Path) -> None:
    stage1_nodes, stage1_init = load_stage(checkpoint_dir / "ap_sdn_stage1_int8.onnx", "s1_", {"input", "hidden1", "exit1"})
    stage2_nodes, stage2_init = load_stage(checkpoint_dir / "ap_sdn_stage2_int8.onnx", "s2_", {"hidden1", "hidden2", "exit2"})
    stage3_nodes, stage3_init = load_stage(checkpoint_dir / "ap_sdn_stage3_int8.onnx", "s3_", {"hidden2", "exit3"})

    exit3_const = exit_point_const(3, "exit3_point")
    depth2_graph = helper.make_graph(
        list(stage3_nodes) + [exit3_const], "depth2", [], [
            helper.make_tensor_value_info("exit3", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("exit3_point", TensorProto.INT64, []),
        ],
        initializer=stage3_init,
    )

    cond2_nodes, cond2 = confidence_and_cond_nodes("exit2", threshold, "s2")
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
    if2_node = helper.make_node("If", [cond2], ["out_logits_2", "out_exit_2"], then_branch=then2_graph, else_branch=depth2_graph)
    depth1_else_graph = helper.make_graph(
        list(stage2_nodes) + cond2_nodes + [exit2_const, if2_node], "depth1_else", [], [
            helper.make_tensor_value_info("out_logits_2", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("out_exit_2", TensorProto.INT64, []),
        ],
        initializer=stage2_init,
    )

    cond1_nodes, cond1 = confidence_and_cond_nodes("exit1", threshold, "s1")
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
    if1_node = helper.make_node("If", [cond1], ["logits", "exit_point"], then_branch=then1_graph, else_branch=depth1_else_graph)
    main_nodes = list(stage1_nodes) + cond1_nodes + [exit1_const, if1_node]
    main_graph = helper.make_graph(
        main_nodes, "sdn_unified_int8",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, INPUT_SIZE])],
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
    parser = argparse.ArgumentParser(description="Build INT8 unified (If-node) SDN-style ONNX graph.")
    parser.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--skip-stage-quantize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_stage_quantize:
        quantize_stages(args.checkpoint_dir)
    build(args.checkpoint_dir, args.confidence_threshold, args.checkpoint_dir / "ap_sdn_unified_int8.onnx")


if __name__ == "__main__":
    main()
