"""
generate_from_real_dataset.py
────────────────────────────────────────────────────────────────────
실제 네트워크 데이터셋 → 프로젝트 형식(data/real/) 변환 스크립트

[사용 목적]
  traffic_simulator.py 가 임의 생성한 데이터 대신,
  실제 공개 데이터셋을 동일한 CSV 형식으로 변환합니다.
  dataloader.py, train.py 등 기존 코드 수정 없이 그대로 사용 가능합니다.

[지원 데이터셋 프리셋] (--dataset 옵션)
  kaggle_6g     : Kaggle – 6G Network Slicing QoS Dataset (★ 기본 추천)
                  다운로드: kaggle.com/datasets/ziya07/wireless-network-slicing-dataset
                  컬럼: Traffic Load, Bandwidth Utilization, Packet Loss Rate,
                        Latency, Network Slice ID
  custom        : 직접 컬럼 지정 (--col-* 옵션 필수)

[기본 사용 예시]
  python generate_from_real_dataset.py \
      --dataset kaggle_6g \
      --input   project/data/external/6G_network_slicing_qos_dataset.csv \
      --out-dir project/data/real \
      --overwrite-real

[커스텀 모드]
  python generate_from_real_dataset.py \
      --dataset custom \
      --input   project/data/external/my_data.csv \
      --col-rps        traffic_load \
      --col-occupancy  bandwidth_utilization \
      --col-loss       packet_loss_rate \
      --col-latency    latency_ms

[출력 파일] (project/data/real/ 기본)
  train.csv, val.csv, test.csv
  scaler_params.json
  dataset_summary.json
  conversion_report.txt

[GitHub 업로드]
  git add project/data/real/train.csv project/data/real/val.csv \
           project/data/real/test.csv  project/data/real/scaler_params.json \
           project/data/real/dataset_summary.json \
           project/data/real/conversion_report.txt \
           project/scripts/generate_from_real_dataset.py
  git commit -m "feat: 실제 데이터셋(6G Network Slicing QoS) 변환 결과 추가"
  git push origin yena
────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════
#  1. 데이터셋 프리셋
# ══════════════════════════════════════════════════════════════════

DATASET_PRESETS = {

    # ── ★ 기본 추천: 6G Network Slicing QoS Dataset ──────────────
    "kaggle_6g": {
        "description": "Kaggle – 6G Network Slicing QoS Dataset",
        "url": "https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset",
        "citation": (
            "6G Network Slicing QoS Dataset, Kaggle (ziya07). "
            "URL: https://www.kaggle.com/datasets/ziya07/wireless-network-slicing-dataset"
        ),
        # ── 원본 컬럼명 ───────────────────────────────────────────
        # 이 데이터셋의 모든 수치 피처는 이미 0~1로 정규화되어 있음.
        # → 물리적 단위 복원 후 다시 Min-Max 정규화 적용.
        "col_rps":        "Traffic Load (bps)",
        "col_occupancy":  "Bandwidth Utilization (%)",
        "col_loss":       "Packet Loss Rate (%)",
        "col_latency":    "Latency (ms)",
        "timestamp_col":  "Timestamp",
        "scenario_col":   "Network Slice ID",   # 1~4 → 4가지 시나리오로 매핑
        # ── 단위 복원 배율 ────────────────────────────────────────
        # 원본 값이 0~1이므로, 각 피처의 물리적 최댓값을 곱해 단위 복원
        "rps_restore":        1000.0,   # 0~1 → 0~1000 RPS
        "occupancy_restore":   100.0,   # 0~1 → 0~100 %
        "loss_restore":         30.0,   # 0~1 → 0~30 %
        "latency_restore":     500.0,   # 0~1 → 0~500 ms
        # ── 시나리오 매핑 ─────────────────────────────────────────
        # Network Slice ID 1~4 → 프로젝트 시나리오 이름
        # 매핑 근거: 슬라이스 특성(과부하 빈도·트래픽 패턴)과 시나리오 유형 대응
        "scenario_map": {
            1: "startup_surge",        # 슬라이스 1 – 점진적 부하 증가 패턴
            2: "emergency_ramp",       # 슬라이스 2 – 고부하 폭증 패턴
            3: "lunch_restart",        # 슬라이스 3 – 저→중 점진 증가 패턴
            4: "imbalanced_ap_load",   # 슬라이스 4 – 특정 구간 지속 고부하
        },
    },

    # ── 커스텀 모드 (CLI 옵션으로 컬럼 직접 지정) ─────────────────
    "custom": {
        "description": "사용자 지정 데이터셋",
        "url": "(직접 입력)",
        "citation": "(직접 입력)",
        "col_rps":       None,
        "col_occupancy": None,
        "col_loss":      None,
        "col_latency":   None,
        "timestamp_col": None,
        "scenario_col":  None,
        "rps_restore":      1.0,
        "occupancy_restore": 1.0,
        "loss_restore":      1.0,
        "latency_restore":   1.0,
        "scenario_map":  {},
    },
}

# ══════════════════════════════════════════════════════════════════
#  2. 레이블 기준 (channel_occupancy 기반 – 기존 프로젝트 정의 동일)
# ══════════════════════════════════════════════════════════════════

LABEL_THRESHOLDS = [
    (0,  0.0,  40.0),   # 정상
    (1, 40.0,  65.0),   # 혼잡 경고
    (2, 65.0,  85.0),   # 혼잡
    (3, 85.0, 100.1),   # 심각 혼잡
]
LABEL_NAMES = {0: "정상", 1: "혼잡_경고", 2: "혼잡", 3: "심각_혼잡"}

# ══════════════════════════════════════════════════════════════════
#  3. 정규화 기준값 (기존 scaler_params.json 과 동일)
# ══════════════════════════════════════════════════════════════════

SCALER_PARAMS = {
    "rps":               {"min": 0.0,  "max": 1000.0},
    "channel_occupancy": {"min": 0.0,  "max": 100.0},
    "packet_loss":       {"min": 0.0,  "max": 30.0},
    "latency":           {"min": 0.0,  "max": 500.0},
}

WINDOW_SIZE  = 10
STRIDE       = 1
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42


# ══════════════════════════════════════════════════════════════════
#  4. 피처 변환
# ══════════════════════════════════════════════════════════════════

def build_feature_df(raw: pd.DataFrame, preset: dict, args) -> pd.DataFrame:
    """
    원본 DataFrame → 프로젝트 4개 피처 DataFrame 변환

    핵심 로직 (kaggle_6g 기준):
      원본 값이 0~1 정규화 상태이므로, 물리적 최댓값을 곱해 단위 복원 후
      이후 파이프라인에서 다시 Min-Max 정규화를 적용함.
      (복원 → 재정규화 = 원본 0~1 값 그대로 유지되는 항등 변환이지만,
       레이블 생성 시 channel_occupancy 가 0~100% 스케일이어야 하므로
       이 단계에서 ×100 복원이 반드시 필요함)
    """
    df = pd.DataFrame()

    # 커스텀 모드: CLI 옵션으로 컬럼명 덮어쓰기
    col_rps = args.col_rps or preset["col_rps"]
    col_occ = args.col_occupancy or preset["col_occupancy"]
    col_loss = args.col_loss or preset["col_loss"]
    col_lat  = args.col_latency or preset["col_latency"]

    for col, fname in [(col_rps, "rps"), (col_occ, "channel_occupancy"),
                       (col_loss, "packet_loss"), (col_lat, "latency")]:
        if col not in raw.columns:
            candidates = [c for c in raw.columns
                          if any(k in c.lower() for k in fname.lower().split("_"))]
            raise KeyError(
                f"\n[오류] '{fname}' 매핑 컬럼 '{col}' 을 찾을 수 없습니다.\n"
                f"  전체 컬럼: {list(raw.columns)}\n"
                + (f"  유사 후보: {candidates}\n" if candidates else "")
                + f"  힌트: --col-{fname.replace('_','-')} <컬럼명>\n"
            )

    # 물리적 단위 복원 (0~1 → 원래 범위)
    df["rps"]               = raw[col_rps]  * preset["rps_restore"]
    df["channel_occupancy"] = raw[col_occ]  * preset["occupancy_restore"]
    df["packet_loss"]       = raw[col_loss] * preset["loss_restore"]
    df["latency"]           = raw[col_lat]  * preset["latency_restore"]

    # 결측값 처리 (시계열이므로 ffill 우선)
    df = df.ffill().bfill().fillna(0)

    # 범위 클리핑 (데이터 이상치 보정)
    df["rps"]               = df["rps"].clip(0, 1000)
    df["channel_occupancy"] = df["channel_occupancy"].clip(0, 100)
    df["packet_loss"]       = df["packet_loss"].clip(0, 30)
    df["latency"]           = df["latency"].clip(0, 500)

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
#  5. 레이블 생성 (channel_occupancy 기반)
# ══════════════════════════════════════════════════════════════════

def assign_labels(channel_occupancy: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=channel_occupancy.index)
    for lbl, lo, hi in LABEL_THRESHOLDS:
        labels[(channel_occupancy >= lo) & (channel_occupancy < hi)] = lbl
    return labels


# ══════════════════════════════════════════════════════════════════
#  6. 시나리오 할당
# ══════════════════════════════════════════════════════════════════

SCENARIO_NAMES = [
    "startup_surge", "emergency_ramp", "lunch_restart", "imbalanced_ap_load"
]


def assign_scenarios(raw: pd.DataFrame, preset: dict) -> pd.Series:
    sc_col = preset.get("scenario_col")
    sc_map = preset.get("scenario_map", {})

    # 원본 시나리오 컬럼이 있고 매핑 딕셔너리도 있으면 직접 매핑
    if sc_col and sc_col in raw.columns and sc_map:
        mapped = raw[sc_col].map(sc_map)
        if mapped.isna().sum() == 0:
            print(f"      '{sc_col}' 컬럼 → 시나리오 직접 매핑 완료")
            return mapped.reset_index(drop=True).astype(str)
        print(f"      [경고] 일부 {sc_col} 값 매핑 실패 → 순환 배정으로 대체")

    # 시나리오 컬럼 없음 → 인덱스 기반 순환 배정 (4가지 균등)
    print("      시나리오 컬럼 없음 → 순환 배정")
    return pd.Series([SCENARIO_NAMES[i % 4] for i in range(len(raw))],
                     dtype=str)


# ══════════════════════════════════════════════════════════════════
#  7. 슬라이딩 윈도우
# ══════════════════════════════════════════════════════════════════

def apply_sliding_window(feat_df: pd.DataFrame,
                          labels: pd.Series,
                          scenarios: pd.Series) -> pd.DataFrame:
    rows = []
    n = len(feat_df)
    for sample_id, start in enumerate(range(0, n - WINDOW_SIZE + 1, STRIDE)):
        end = start + WINDOW_SIZE
        window = feat_df.iloc[start:end]
        label    = int(labels.iloc[end - 1])
        scenario = str(scenarios.iloc[start])
        for ts in range(WINDOW_SIZE):
            rows.append({
                "sample_id":         sample_id,
                "timestep":          ts,
                "rps":               window.iloc[ts]["rps"],
                "channel_occupancy": window.iloc[ts]["channel_occupancy"],
                "packet_loss":       window.iloc[ts]["packet_loss"],
                "latency":           window.iloc[ts]["latency"],
                "label":             label,
                "scenario":          scenario,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#  8. Min-Max 정규화 (고정 기준값)
# ══════════════════════════════════════════════════════════════════

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feat, p in SCALER_PARAMS.items():
        df[feat] = ((df[feat] - p["min"]) / (p["max"] - p["min"])).clip(0.0, 1.0)
    return df


# ══════════════════════════════════════════════════════════════════
#  9. Stratified Split (sample_id 단위)
# ══════════════════════════════════════════════════════════════════

def stratified_split(windowed: pd.DataFrame):
    meta = windowed.groupby("sample_id")["label"].last().reset_index()

    train_ids, temp_ids = train_test_split(
        meta["sample_id"],
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=meta["label"],
        random_state=RANDOM_SEED,
    )
    temp_meta = meta[meta["sample_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_meta["sample_id"],
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_meta["label"],
        random_state=RANDOM_SEED,
    )

    def reindex(df):
        df = df.copy()
        id_map = {old: new for new, old in enumerate(df["sample_id"].unique())}
        df["sample_id"] = df["sample_id"].map(id_map)
        return df

    return (
        reindex(windowed[windowed["sample_id"].isin(train_ids)]),
        reindex(windowed[windowed["sample_id"].isin(val_ids)]),
        reindex(windowed[windowed["sample_id"].isin(test_ids)]),
    )


# ══════════════════════════════════════════════════════════════════
#  10. 리포트 생성
# ══════════════════════════════════════════════════════════════════

def build_summary(train_df, val_df, test_df, source_info: dict) -> dict:
    def label_dist(df):
        c = df.drop_duplicates("sample_id")["label"].value_counts().sort_index()
        return {LABEL_NAMES[k]: int(v) for k, v in c.items()}

    def scenario_dist(df):
        c = df.drop_duplicates("sample_id")["scenario"].value_counts()
        return {k: int(v) for k, v in c.items()}

    total = sum(len(d) // WINDOW_SIZE for d in [train_df, val_df, test_df])
    return {
        "source": source_info,
        "total_samples": total,
        "window_size": WINDOW_SIZE,
        "features": list(SCALER_PARAMS.keys()),
        "label_thresholds": {
            f"label_{l}": f"{lo:.0f}% ~ {hi:.0f}%"
            for l, lo, hi in LABEL_THRESHOLDS
        },
        "scaler_params": SCALER_PARAMS,
        "splits": {
            split: {
                "samples": len(df) // WINDOW_SIZE,
                "rows":    len(df),
                "shape":   f"({len(df)//WINDOW_SIZE}, {WINDOW_SIZE}, 4)",
                "label_distribution":    label_dist(df),
                "scenario_distribution": scenario_dist(df),
            }
            for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]
        },
    }


def write_report(path: Path, summary: dict, preset: dict, warnings_list: list):
    W = 65
    lines = [
        "=" * W,
        "  데이터셋 변환 리포트  –  generate_from_real_dataset.py",
        "=" * W,
        "",
        f"  데이터셋  : {summary['source']['description']}",
        f"  출처 URL  : {summary['source']['url']}",
        f"  입력 파일 : {summary['source']['input_file']}",
        f"  원본 행수 : {summary['source']['raw_rows']:,}",
        "",
        "  ── 출처 표기 (보고서/발표 자료용) ──",
        f"  {preset.get('citation', preset.get('description', ''))}",
        "",
        f"  윈도우 크기 : {summary['window_size']}  (슬라이딩, 스트라이드=1)",
        "  정규화      : Min-Max (고정 기준값)",
        f"  분할        : Train {int(TRAIN_RATIO*100)}% / "
        f"Val {int(VAL_RATIO*100)}% / Test {int(TEST_RATIO*100)}%",
        "",
        "  ── 레이블 기준 (channel_occupancy) ──",
        "  ┌──────┬──────────────┬─────────────┐",
        "  │ 레이블 │ 혼잡 수준    │ 채널 점유율 │",
        "  ├──────┼──────────────┼─────────────┤",
        "  │   0  │ 정상         │  0 ~ 40 %   │",
        "  │   1  │ 혼잡 경고    │ 40 ~ 65 %   │",
        "  │   2  │ 혼잡         │ 65 ~ 85 %   │",
        "  │   3  │ 심각 혼잡    │ 85 ~ 100 %  │",
        "  └──────┴──────────────┴─────────────┘",
        "",
        f"  전체 샘플 수 : {summary['total_samples']:,}",
        "",
    ]
    for split, info in summary["splits"].items():
        lines.append(
            f"  [{split.upper():5}]  {info['samples']:>5}샘플 / "
            f"{info['rows']:>6}행 / shape {info['shape']}"
        )
        for lbl_name, cnt in info["label_distribution"].items():
            lines.append(f"    {lbl_name:15}: {cnt:>5}")
        sc = info["scenario_distribution"]
        lines.append(f"    시나리오 분포  : " + ", ".join(f"{k}:{v}" for k, v in sc.items()))
        lines.append("")

    if warnings_list:
        lines += ["  ⚠ 경고", "  " + "-" * 50]
        for w in warnings_list:
            lines.append(f"  - {w}")
        lines.append("")
    lines.append("=" * W)

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print(text)


# ══════════════════════════════════════════════════════════════════
#  11. CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="실제 네트워크 데이터셋 → 프로젝트 형식 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()),
                   default="kaggle_6g",
                   help="데이터셋 프리셋 (기본: kaggle_6g)")
    p.add_argument("--input", "-i", required=True,
                   help="입력 CSV 경로")
    p.add_argument("--out-dir", "-o", default="project/data/real",
                   help="출력 디렉토리 (기본: project/data/real)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--min-label-samples", type=int, default=10,
                   help="레이블별 최소 샘플 수 (미달 시 경고)")
    p.add_argument("--col-rps",       default=None)
    p.add_argument("--col-occupancy", default=None)
    p.add_argument("--col-loss",      default=None)
    p.add_argument("--col-latency",   default=None)
    p.add_argument("--overwrite-real", action="store_true",
                   help="기존 data/real/ 덮어쓰기 확인 생략")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
#  12. 메인
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    global RANDOM_SEED
    RANDOM_SEED = args.seed

    out_dir = Path(args.out_dir)

    # 덮어쓰기 확인
    if out_dir.exists() and any(out_dir.glob("*.csv")) and not args.overwrite_real:
        print(f"\n[주의] '{out_dir}' 에 기존 CSV가 있습니다. 덮어쓰려면 y 입력: ", end="")
        if input().strip().lower() != "y":
            print("취소.")
            sys.exit(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    preset = DATASET_PRESETS[args.dataset]

    # 커스텀 모드 컬럼 검증
    if args.dataset == "custom":
        missing = [f for f, v in [
            ("--col-rps", args.col_rps), ("--col-occupancy", args.col_occupancy),
            ("--col-loss", args.col_loss), ("--col-latency", args.col_latency),
        ] if not v]
        if missing:
            print(f"[오류] custom 모드 필수 옵션 누락: {missing}")
            sys.exit(1)

    print(f"\n[1/7] 데이터셋 로드: {args.input}")
    print(f"      프리셋: {preset['description']}")

    raw = pd.read_csv(args.input, low_memory=False)
    print(f"      원본 행수: {len(raw):,}  컬럼: {list(raw.columns)}")

    # 타임스탬프 정렬
    ts_col = preset.get("timestamp_col") or getattr(args, "timestamp_col", None)
    if ts_col and ts_col in raw.columns:
        raw = raw.sort_values(ts_col).reset_index(drop=True)
        print(f"      '{ts_col}' 기준 시간순 정렬 완료")

    # [2] 피처 변환
    print("\n[2/7] 피처 변환 (단위 복원) ...")
    feat_df = build_feature_df(raw, preset, args)
    for feat in ["rps", "channel_occupancy", "packet_loss", "latency"]:
        mn, mx = feat_df[feat].min(), feat_df[feat].max()
        print(f"      {feat:20}: [{mn:.2f} ~ {mx:.2f}]")

    # [3] 레이블 생성
    print("\n[3/7] 레이블 생성 (channel_occupancy 기준) ...")
    labels = assign_labels(feat_df["channel_occupancy"])
    warnings_list = []
    for lbl, lo, hi in LABEL_THRESHOLDS:
        cnt = int((labels == lbl).sum())
        pct = cnt / len(labels) * 100
        print(f"      레이블 {lbl} ({LABEL_NAMES[lbl]:10}): {cnt:>5,} ({pct:.1f}%)")
        if cnt < args.min_label_samples * WINDOW_SIZE:
            warnings_list.append(
                f"레이블 {lbl}({LABEL_NAMES[lbl]}) 샘플 부족 ({cnt}행) – "
                f"클래스 가중치 적용 권장"
            )

    # [4] 시나리오 할당
    print("\n[4/7] 시나리오 할당 ...")
    scenarios = assign_scenarios(raw, preset)
    for sc, cnt in scenarios.value_counts().items():
        print(f"      {sc}: {cnt:,}")

    # [5] 슬라이딩 윈도우
    print(f"\n[5/7] 슬라이딩 윈도우 (크기={WINDOW_SIZE}, 스트라이드={STRIDE}) ...")
    windowed = apply_sliding_window(feat_df, labels, scenarios)
    n_samples = len(windowed) // WINDOW_SIZE
    print(f"      총 샘플: {n_samples:,}  행: {len(windowed):,}")

    # [6] 정규화
    print("\n[6/7] Min-Max 정규화 ...")
    windowed = normalize(windowed)

    # [7] Split & 저장
    print(f"\n[7/7] Stratified Split & 저장 → {out_dir}/")
    train_df, val_df, test_df = stratified_split(windowed)

    train_df.to_csv(out_dir / "train.csv",  index=False)
    val_df.to_csv(out_dir   / "val.csv",    index=False)
    test_df.to_csv(out_dir  / "test.csv",   index=False)

    (out_dir / "scaler_params.json").write_text(
        json.dumps(SCALER_PARAMS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    source_info = {
        "description": preset["description"],
        "url":         preset.get("url", ""),
        "citation":    preset.get("citation", ""),
        "input_file":  str(args.input),
        "raw_rows":    len(raw),
    }
    summary = build_summary(train_df, val_df, test_df, source_info)
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report(out_dir / "conversion_report.txt", summary, preset, warnings_list)

    print(f"\n✅ 완료! 생성 파일:")
    for f in ["train.csv", "val.csv", "test.csv",
              "scaler_params.json", "dataset_summary.json", "conversion_report.txt"]:
        size = (out_dir / f).stat().st_size
        print(f"   {out_dir}/{f}  ({size/1024:.1f} KB)")

    if warnings_list:
        print("\n⚠ 경고:")
        for w in warnings_list:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
