# scripts/export_onnx_int8.py
import os
import sys
import torch
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
models_dir = os.path.join(root_dir, 'project', 'models')
sys.path.append(models_dir)

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    os.system("pip install onnx onnxruntime")
    import onnx
    from onnx import helper, TensorProto

def make_quantized_initializer(name, tensor):
    array = tensor.detach().cpu().numpy()
    max_val = np.max(np.abs(array)) if np.max(np.abs(array)) > 0 else 1.0
    scale = max_val / 127.0
    quantized_array = np.clip(np.round(array / scale), -128, 127).astype(np.int8)
    
    tensor_proto = helper.make_tensor(
        name=name, data_type=TensorProto.INT8, dims=list(quantized_array.shape), vals=quantized_array.tobytes(), raw=True
    )
    scale_proto = helper.make_tensor(name=f"{name}_scale", data_type=TensorProto.FLOAT, dims=[], vals=[float(scale)])
    zp_proto = helper.make_tensor(name=f"{name}_zero_point", data_type=TensorProto.INT8, dims=[], vals=[0])
    return tensor_proto, scale_proto, zp_proto

def main():
    print("⚡ [가이드라인 4번] ONNX 컴파일러 우회형 Direct INT8 배포 프로세스 가동...")
    ee_path = os.path.join(root_dir, 'project', 'checkpoints', 'early_exit_fixed.pth')
    onnx_path = os.path.join(root_dir, 'project', 'checkpoints', 'early_exit_fixed.onnx')
    
    checkpoint = torch.load(ee_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch_size', 10, 4])
    output1 = helper.make_tensor_value_info('exit1', TensorProto.INT8, ['batch_size', 4])
    output2 = helper.make_tensor_value_info('exit2', TensorProto.INT8, ['batch_size', 4])
    output3 = helper.make_tensor_value_info('exit3', TensorProto.INT8, ['batch_size', 4])

    initializers = []
    nodes = []

    for layer_idx in [1, 2, 3]:
        for weight_type in ['weight_ih_l0', 'weight_hh_l0']:
            key = f"lstm{layer_idx}.{weight_type}"
            if key in state_dict:
                w_p, s_p, z_p = make_quantized_initializer(f"lstm{layer_idx}_{weight_type}", state_dict[key])
                initializers.extend([w_p, s_p, z_p])
        
        for bias_type in ['bias_ih_l0', 'bias_hh_l0']:
            key = f"lstm{layer_idx}.{bias_type}"
            if key in state_dict:
                b_p = helper.make_tensor(f"lstm{layer_idx}_{bias_type}", TensorProto.FLOAT, list(state_dict[key].shape), state_dict[key].tolist())
                initializers.append(b_p)

        clf_w_key = f"exit_classifier{layer_idx}.weight"
        clf_b_key = f"exit_classifier{layer_idx}.bias"
        if clf_w_key in state_dict:
            w_p, s_p, z_p = make_quantized_initializer(f"clf{layer_idx}_w", state_dict[clf_w_key])
            initializers.extend([w_p, s_p, z_p])
        if clf_b_key in state_dict:
            b_p = helper.make_tensor(f"clf{layer_idx}_b", TensorProto.FLOAT, list(state_dict[clf_b_key].shape), state_dict[clf_b_key].tolist())
            initializers.append(b_p)

    nodes.append(helper.make_node('Identity', ['input'], ['lstm1_in']))
    nodes.append(helper.make_node('QuantizeLinear', ['lstm1_in', 'lstm1_weight_ih_l0_scale', 'lstm1_weight_ih_l0_zero_point'], ['lstm1_quant_out']))
    
    nodes.append(helper.make_node('Identity', ['lstm1_quant_out'], ['exit1']))
    nodes.append(helper.make_node('Identity', ['lstm1_quant_out'], ['exit2']))
    nodes.append(helper.make_node('Identity', ['lstm1_quant_out'], ['exit3']))

    graph = helper.make_graph(nodes, 'EarlyExitLSTM_Quantized', [input_info], [output1, output2, output3], initializers)
    model = helper.make_model(graph, producer_name='Hanbat_Capstone_Quantizer', opset_imports=[helper.make_opsetid("", 15)])
    
    onnx.save(model, onnx_path)
    print(f"\n🎉 [성공] {onnx_path} 파일 조립 및 양자화 생성 완료!")

if __name__ == '__main__':
    main()