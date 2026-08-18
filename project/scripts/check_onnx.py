# scripts/check_onnx.py
import os
import sys
import onnxruntime as ort
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
onnx_path = os.path.join(root_dir, 'project', 'checkpoints', 'early_exit_fixed.onnx')

print(f"⚡ [가이드라인 4번] ONNX 모델 변환 후 정상 동작 검증 중...")
if not os.path.exists(onnx_path):
    print("❌ 파일이 없습니다. export 스크립트를 먼저 실행하세요.")
    sys.exit(1)

session = ort.InferenceSession(onnx_path)
dummy = np.random.randn(1, 10, 4).astype(np.float32)
result = session.run(None, {'input': dummy})

print("=" * 50)
print("🎉 [정상 출력 확인] ONNX Runtime Engine 추론 성공!")
print(result) 
print("=" * 50)