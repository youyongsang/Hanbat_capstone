"""
comparison_summary.csv 생성 스크립트
장예나 Stage 3 — 4개 방식 비교 결과 취합

수치 출처:
  ① 임계값  → project/results/hojung/comparison_summary.csv
  ② LSTM    → project/results/hojung/comparison_summary.csv
  ③ EE 고정 → project/results/hojung/comparison_summary.csv
              + project/results/yongsang/early_exit_stage2_comparison_report.txt
  ④ EE 동적 → 동일
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project.utils.logger import save_results, print_summary

SAVE_PATH = Path("project/results/yena/comparison_summary.csv")

if SAVE_PATH.exists():
    SAVE_PATH.unlink()

# ① 임계값 방식 — Accuracy 42.2%, Avg 0.0031ms
save_results("threshold",               0.422, 0.0031,
             None,                      SAVE_PATH, unnecessary_switch_rate=0.0)

# ② Baseline LSTM — Accuracy 94.9%, Avg 0.4816ms
save_results("baseline_lstm",           0.949, 0.4816,
             None,                      SAVE_PATH, unnecessary_switch_rate=0.0)

# ③ Early Exit + 고정 θ — Accuracy 95.7%, Avg 0.3903ms
#    Exit 1: 20.5% / Exit 2: 71.8% / Exit 3: 7.7%
save_results("early_exit_fixed_theta",  0.957, 0.3903,
             {1: 0.205, 2: 0.718, 3: 0.077}, SAVE_PATH, unnecessary_switch_rate=0.0)

# ④ Early Exit + 동적 θ — Accuracy 96.3%, Avg 0.4053ms
#    Exit 1: 25.6% / Exit 2: 69.5% / Exit 3: 4.8%
save_results("early_exit_dynamic_theta", 0.963, 0.4053,
             {1: 0.256, 2: 0.695, 3: 0.048}, SAVE_PATH, unnecessary_switch_rate=0.0)

print(f"저장 완료: {SAVE_PATH}")
print()
print_summary(SAVE_PATH)
