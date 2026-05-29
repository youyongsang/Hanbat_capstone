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
        print(f"[ERROR] ONNX file not found: {onnx_path}")
        return

    print("[Multi-head Confidence Filtering] runtime simulator start")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 조기 종료 확신도 기준치 (85%)
    THRESHOLD = 0.85
    
    times = []
    exit_counts = {1: 0, 2: 0, 3: 0}
    
    print("Run 100 streaming inference trials...")
    
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
    print("[OK] Runtime-controlled Early Exit report")
    print("="*50)
    print(f"Average inference latency : {np.mean(times):.3f} ms")
    print(f"Min: {np.min(times):.3f} ms / Max: {np.max(times):.3f} ms")
    print("-"*50)
    print("Pseudo Early Exit filtering result:")
    print(f"  Exit 1: {exit_counts[1]}")
    print(f"  Exit 2: {exit_counts[2]}")
    print(f"  Exit 3: {exit_counts[3]}")
    print("-"*50)
    print(f"Final logits example from last inference: {final_logits.shape}")
    print(f"   {final_logits}")
    print("="*50)

if __name__ == '__main__':
    main()
