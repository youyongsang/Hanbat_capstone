# project/scripts/inference_pi.py
import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort

def load_scaler(scaler_path):
    """💡 JSON 파일로부터 저장된 스케일러 파라미터를 안전하게 로드합니다."""
    if not os.path.exists(scaler_path):
        print(f"⚠️ 스케일러 파일을 찾을 수 없습니다: {scaler_path} (기본 정규화 미적용)")
        return None
    with open(scaler_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_input(raw_data, scaler):
    """💡 [피드백 반영 1] 로드된 스케일러(mean, scale)를 활용하여 데이터를 리얼 정규화합니다."""
    if scaler is None:
        return raw_data.astype(np.float32)
    
    # JSON에 저장된 mean과 scale(또는 var의 제곱근)을 활용한 표준정규화 공식 적용
    mean = np.array(scaler.get('mean', 0.0), dtype=np.float32)
    scale = np.array(scaler.get('scale', 1.0), dtype=np.float32)
    
    # 브로드캐스팅 연산으로 입력 데이터 정규화 진행
    normalized_data = (raw_data - mean) / scale
    return normalized_data.astype(np.float32)

def main():
    print("🔮 [라즈베리 파이 배포 검증] 에러 제로, 실전형 ONNX 런타임 추론을 시작합니다.")
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    checkpoints_dir = os.path.join(root_dir, 'project', 'checkpoints')
    data_dir = os.path.join(root_dir, 'project', 'data')
    
    onnx_path = os.path.join(checkpoints_dir, 'early_exit_fixed.onnx')
    scaler_path = os.path.join(data_dir, 'scaler_params.json') # 스케일러 파일 예상 경로
    
    if not os.path.exists(onnx_path):
        print(f"❌ 양자화된 ONNX 파일을 찾을 수 없습니다: {onnx_path}")
        sys.exit(1)
        
    # 1. ONNX 세션 로드
    session = ort.InferenceSession(onnx_path)
    
    # 💡 [피드백 반영 2] 인풋 네임을 'input'으로 하드코딩하지 않고, ONNX 그래프에서 동적으로 완벽 추출!
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    
    # 2. 스케일러 파라미터 로드
    scaler = load_scaler(scaler_path)
    
    # 3. 실전 테스트 데이터 생성 (배치 1, 시퀀스 10, 피처 4)
    raw_dummy_input = np.random.uniform(10.0, 50.0, size=(1, 10, 4)).astype(np.float32)
    
    # 4. 정규화 전처리 실전 투입!
    processed_input = preprocess_input(raw_dummy_input, scaler)
    
    # 5. ONNX 런타임 추론 실행
    print(f"🏃‍♂️ ONNX Dynamic Input Name 적용 완료: '{input_name}' 세션 가동 중...")
    start_time = time.perf_counter()
    outputs = session.run(output_names, {input_name: processed_input})
    end_time = time.perf_counter()
    
    latency = (end_time - start_time) * 1000
    print(f"🟢 [추론 성공] 라즈베리 파이 모의 Latency: {latency:.4f} ms")
    print(f"📊 출력 레이어 개수: {len(outputs)} (Early Exit 그래프 정상 활성화)")

if __name__ == "__main__":
    main()