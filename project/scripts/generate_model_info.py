"""
model_info.json 생성 스크립트
유용상 Stage 3 — 최종 모델 파라미터 기록

용상이 실제 튜닝을 마친 뒤 최종 수치로 교체하여 실행.
결과 파일: project/checkpoints/model_info.json

현재 값 = 호중 최종 4개 방식 비교 결과 기준
"""

import json
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("project/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CHECKPOINT_DIR / "model_info.json"

# ── 최종 비교 실험 기준 파라미터 ────────────────────────────────────────

model_info = {
    # ── 모델 구조 ──────────────────────────────────────────────────────────
    "hidden_size":  128,
    "num_layers":   3,
    "dropout":      0.2,
    "num_classes":  4,

    # ── 고정 θ (Baseline ③) ───────────────────────────────────────────────
    "fixed_theta_1": 0.3,
    "fixed_theta_2": 0.6,

    # ── 동적 θ 파라미터 (제안 모델 ④) ───────────────────────────────────
    "dynamic_base_theta_1":  0.3,
    "dynamic_base_theta_2":  0.6,
    "dynamic_high_variance": 0.22,    # 정규화 스케일 0~1 기준
    "dynamic_mid_variance":  0.12,
    "dynamic_min_threshold": 0.22,
    "dynamic_recent_steps":  5,
    "dynamic_spike_threshold": 0.25,

    # ── 학습 결과 ─────────────────────────────────────────────────────────
    "best_val_accuracy": 0.947,       # Stage 2 학습 최고 val accuracy

    # ── 최종 테스트 성능 (호중 최종 4개 방식 비교 기준) ────────────────
    "test_accuracy_fixed":   0.974359,
    "test_accuracy_dynamic": 0.974359,

    # ── Stage 3 튜닝 메모 ─────────────────────────────────────────────────
    "_note": (
        "호중 최종 4개 방식 비교 결과와 양자화 평가 기준을 맞춘 값. "
        "Early Exit 평가는 theta_1=0.3, theta_2=0.6 기준으로 수행한다."
    ),

    # ── 체크포인트 파일명 ─────────────────────────────────────────────────
    "checkpoints": {
        "best":    "early_exit_lstm_best.pth",
        "fixed":   "early_exit_fixed.pth",
        "dynamic": "early_exit_dynamic.pth",
    },
}

with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f"model_info.json 저장: {OUT_PATH}")
print()
print(json.dumps(model_info, ensure_ascii=False, indent=2))
