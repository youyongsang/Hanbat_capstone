import onnxruntime as ort
import numpy as np
import os
import time

def softmax(x):
    """안정적인 확률 변환을 위한 Softmax (Overflow 방지)"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def main():
    onnx_path = 'project/checkpoints/early_exit_fixed.onnx'
    
    if not os.path.exists(onnx_path):
        print(f"❌ [에러] ONNX 파일이 없습니다: {onnx_path}")
        return

    print(f"🚀 [Multi-head Confidence Filtering] 실전 런타임 시뮬레이터 가동...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 조기 종료 확신도 기준치 (85%)
    THRESHOLD = 0.85
    
    times = []
    exit_counts = {1: 0, 2: 0, 3: 0}
    
    print(f"📊 총 100번의 스트리밍 데이터 추론 테스트 시작...")
    
    for _ in range(100):
        dummy = np.random.randn(1, 10, 4).astype(np.float32)
        
        start = time.time()
        # 1. 멀티 헤드 출력 그래프 전체 추론 (PC/라즈베리파이 공통 초고속 연산)
        outputs = session.run(None, {'input': dummy})
        
        # 기본값은 마지막 3단계 결과로 세팅
        chosen_exit = 3
        final_logits = outputs[2][0].astype(np.float32)
        
        # 2. ⚡ 런타임 레벨 조기 종료 선택 (Confidence Filtering)
        for exit_idx in range(3):
            # 피드백 반영: 타입 안정성 확보 및 오버플로우 방지
            logits = outputs[exit_idx][0].astype(np.float32)
            probs = softmax(logits)
            max_prob = np.max(probs)
            
            # 확신도가 기준치를 넘으면 해당 레이어에서 즉시 최종 결과 락온(Lock-on) 후 탈출
            if max_prob >= THRESHOLD:
                chosen_exit = exit_idx + 1
                final_logits = logits  # 피드백 반영: 실제 사용될 최종 결과물 저장
                break
                
        inference_time = (time.time() - start) * 1000
        times.append(inference_time)
        exit_counts[chosen_exit] += 1

    print("\n" + "="*50)
    print(f"🎉 [검증 완료] 런타임 제어형 Early Exit 최종 레포트")
    print("="*50)
    print(f"⏱️ 평균 추론 지연 시간 : {np.mean(times):.3f} ms")
    print(f"⏱️ 최소: {np.min(times):.3f} ms / 최대: {np.max(times):.3f} ms")
    print("-"*50)
    print(f"💡 의사 조기 종료(Pseudo Early Exit) 필터링 결과:")
    print(f" 🔹 Exit 1 확정 탈출 : {exit_counts[1]}회")
    print(f" 🔹 Exit 2 확정 탈출 : {exit_counts[2]}회")
    print(f" 🔹 Exit 3 최종 통과 : {exit_counts[3]}회")
    print("-"*50)
    print(f"📦 최종 저장된 Logits 예시 (마지막 추론): {final_logits.shape}")
    print(f"   {final_logits}")
    print("="*50)

if __name__ == '__main__':
    main()