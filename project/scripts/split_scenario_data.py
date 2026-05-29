"""
시나리오별 분석 데이터 분리 스크립트
장예나 Stage 3 — 유용상 Stage 3 분석 지원용

생성 파일:
  project/results/scenario_analysis/
    ├── scenario_0_gradual.csv       # startup_surge (일과 시작)
    ├── scenario_1_spike.csv         # emergency_ramp (긴급 증산)
    ├── scenario_2_periodic.csv      # lunch_restart (점심 재가동)
    └── scenario_3_imbalance.csv     # imbalanced_ap_load (불균형 부하)

  project/results/yongsang/
    └── scenario_analysis_template.csv   # 유용상 모델 예측 입력용 템플릿
"""

import csv
import statistics
from pathlib import Path
from collections import defaultdict

# ── 경로 설정 ──────────────────────────────────────────────────────────────
INPUT_CSV   = Path("project/data/real/test_with_scenario.csv")
OUT_DIR     = Path("project/results/scenario_analysis")
YONA_DIR    = Path("project/results/yongsang")
OUT_DIR.mkdir(parents=True, exist_ok=True)
YONA_DIR.mkdir(parents=True, exist_ok=True)

# ── 시나리오 매핑 ──────────────────────────────────────────────────────────
SCENARIO_MAP = {
    "startup_surge":      ("scenario_0_gradual.csv",    0),
    "emergency_ramp":     ("scenario_1_spike.csv",      1),
    "lunch_restart":      ("scenario_2_periodic.csv",   2),
    "imbalanced_ap_load": ("scenario_3_imbalance.csv",  3),
}

# ── CSV 읽기 ───────────────────────────────────────────────────────────────
print(f"[1/4] {INPUT_CSV} 읽는 중...")
all_rows: list[dict] = []
with INPUT_CSV.open(encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

# 시나리오별 분리
scenario_rows: dict[str, list[dict]] = defaultdict(list)
for row in all_rows:
    scenario_rows[row["scenario"]].append(row)

# ── 파일 1~4: 전체 행 분리 (Full timestep rows) ──────────────────────────
print("[2/4] 시나리오별 전체 행 분리 저장...")
fieldnames = list(all_rows[0].keys())

for sc_name, (filename, sc_id) in SCENARIO_MAP.items():
    rows = scenario_rows[sc_name]
    out_path = OUT_DIR / filename
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    samples = len({r["sample_id"] for r in rows})
    print(f"  {filename}: {samples}샘플 ({len(rows)}행) → {out_path}")

# ── 파일 5: 유용상 분석 템플릿 (샘플 단위, 예측 플레이스홀더 포함) ─────────
print("[3/4] 유용상 분석 템플릿 생성...")

# 샘플 단위로 집계 (true_label + channel_occupancy_variance 계산)
sample_data: dict[tuple, dict] = {}   # (sample_id, scenario) → info

for row in all_rows:
    key = (row["sample_id"], row["scenario"])
    if key not in sample_data:
        sample_data[key] = {
            "sample_id": row["sample_id"],
            "scenario":  row["scenario"],
            "scenario_id": row["scenario_id"],
            "true_label": row["label"],        # 마지막 timestep 기준 레이블
            "occupancies": [],
        }
    sample_data[key]["occupancies"].append(float(row["channel_occupancy"]))
    # 마지막 timestep의 label을 true_label로 사용 (원본과 동일한 기준)
    sample_data[key]["true_label"] = row["label"]

# variance 계산 및 템플릿 CSV 생성
template_fieldnames = [
    "sample_id", "scenario", "scenario_id", "true_label",
    "channel_occupancy_variance",
    # 유용상이 채울 예측 컬럼
    "fixed_pred", "fixed_exit_point", "fixed_theta_1",
    "dynamic_pred", "dynamic_exit_point", "dynamic_theta_1",
]

template_path = YONA_DIR / "scenario_analysis_template.csv"
with template_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=template_fieldnames)
    writer.writeheader()
    for key, info in sorted(sample_data.items(), key=lambda x: (int(x[1]["scenario_id"]), int(x[0][0]))):
        occs = info["occupancies"]
        variance = statistics.stdev(occs) if len(occs) > 1 else 0.0
        writer.writerow({
            "sample_id": info["sample_id"],
            "scenario":  info["scenario"],
            "scenario_id": info["scenario_id"],
            "true_label": info["true_label"],
            "channel_occupancy_variance": round(variance, 6),
            "fixed_pred":     "",   # 유용상이 채움
            "fixed_exit_point": "",
            "fixed_theta_1":  "",
            "dynamic_pred":   "",
            "dynamic_exit_point": "",
            "dynamic_theta_1": "",
        })

print(f"  템플릿 저장: {template_path}")
print(f"  총 {len(sample_data)}샘플")

# ── 결과 요약 ──────────────────────────────────────────────────────────────
print("\n[4/4] 완료 요약")
print("=" * 55)
total = 0
for sc_name, (filename, sc_id) in SCENARIO_MAP.items():
    rows = scenario_rows[sc_name]
    samples = len({r["sample_id"] for r in rows})
    labels = {}
    for r in rows:
        if r["timestep"] == "9":   # 마지막 timestep의 label
            lbl = r["label"]
            labels[lbl] = labels.get(lbl, 0) + 1
    label_str = ", ".join(f"L{k}:{v}" for k, v in sorted(labels.items()))
    print(f"  scenario {sc_id} ({sc_name})")
    print(f"    → {filename}")
    print(f"    샘플 {samples}개 | 레이블 분포: {label_str}")
    total += samples
print(f"\n  전체 테스트 샘플: {total}개")
print("=" * 55)
