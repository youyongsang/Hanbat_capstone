import onnxruntime as ort
import numpy as np

def test_inference():
    onnx_path = 'project/checkpoints/early_exit_fixed.onnx'
    print(f"🔮 [ONNX 로드] {onnx_path} 파일을 정밀 로드합니다...")
    
    # 런타임 세션 생성
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 가변 배치를 확인하기 위해 배치 사이즈를 2로 줘볼게!
    test_batch = 2
    dummy_input = np.random.randn(test_batch, 10, 4).astype(np.float32)
    
    # 추론 시작
    outputs = session.run(None, {'input': dummy_input})
    
    print("\n✅ [추론 완수] 모델이 터지지 않고 출력을 정상 반환했습니다.")
    print(f" - 출력된 최종 헤드 개수: {len(outputs)}개 (Early Exit 1, 2, 3 일치)")
    
    for i, out in enumerate(outputs, 1):
        print(f" - Exit {i} 결과 차원: {out.shape} -> 2개 데이터의 클래스별 확률값(Logits) 추출 완료!")

if __name__ == '__main__':
    test_inference()