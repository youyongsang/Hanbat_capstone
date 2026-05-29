"""test.csv → test_with_scenario.csv 생성 스크립트.

기존 test.csv의 string scenario 컬럼을 유지하면서
정수 scenario_id 컬럼(0~3)을 추가한 test_with_scenario.csv를 생성한다.

시나리오 매핑
-----------
startup_surge      → 0  (일과 시작, 점진적 급증)
emergency_ramp     → 1  (긴급 증산, 갑작스러운 폭증)
lunch_restart      → 2  (점심 재가동, 주기적 패턴)
imbalanced_ap_load → 3  (불균형 부하, 지속적 혼잡)

실행 방법
--------
python project/scripts/generate_test_with_scenario.py
python project/scripts/generate_test_with_scenario.py --data-dir project/data/real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 가이드라인 guideline_yena_stage3.md 기준 시나리오 매핑
SCENARIO_ID_MAP: dict[str, int] = {
    "startup_surge": 0,       # 일과 시작 (점진적 급증)
    "emergency_ramp": 1,      # 긴급 증산 (갑작스러운 폭증)
    "lunch_restart": 2,       # 점심 재가동 (주기적 패턴)
    "imbalanced_ap_load": 3,  # 불균형 부하 (지속적 혼잡)
}

SCENARIO_KR: dict[int, str] = {
    0: "일과 시작 (점진적 급증)",
    1: "긴급 증산 (갑작스러운 폭증)",
    2: "점심 재가동 (주기적 패턴)",
    3: "불균형 부하 (지속적 혼잡)",
}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="test.csv → test_with_scenario.csv (scenario_id 컬럼 추가)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real",
        help="train/val/test.csv가 위치한 디렉토리 (기본: project/data/real)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.data_dir / "test.csv"
    dst = args.data_dir / "test_with_scenario.csv"

    # ── 입력 파일 확인 ──────────────────────────────────────────
    if not src.exists():
        print(f"[오류] 파일 없음: {display_path(src)}")
        sys.exit(1)

    df = pd.read_csv(src)

    if "scenario" not in df.columns:
        print(
            "[오류] test.csv에 'scenario' 컬럼이 없습니다.\n"
            "       generate_real_data.py 또는 generate_from_real_dataset.py로\n"
            "       데이터를 먼저 생성하세요."
        )
        sys.exit(1)

    # ── scenario_id 컬럼 추가 ────────────────────────────────────
    # 기존 string scenario 컬럼을 보존하고 정수 scenario_id를 추가한다.
    # 이유: string 이름은 시나리오 분석 출력에서 가독성에 유리하고,
    #       integer id는 groupby·필터링에 편리하다.
    df["scenario_id"] = df["scenario"].map(SCENARIO_ID_MAP)

    unmapped_mask = df["scenario_id"].isna()
    if unmapped_mask.any():
        unknown = df.loc[unmapped_mask, "scenario"].unique().tolist()
        print(f"[경고] 매핑되지 않은 시나리오 값 발견: {unknown}")
        print("       SCENARIO_ID_MAP을 확인하세요.")

    df["scenario_id"] = df["scenario_id"].astype("Int64")

    # ── 컬럼 순서 정리: scenario_id를 scenario 바로 뒤에 배치 ────
    cols = list(df.columns)
    if "scenario_id" in cols and "scenario" in cols:
        cols.remove("scenario_id")
        scenario_pos = cols.index("scenario")
        cols.insert(scenario_pos + 1, "scenario_id")
        df = df[cols]

    # ── 저장 ─────────────────────────────────────────────────────
    df.to_csv(dst, index=False)

    # ── 결과 요약 출력 ────────────────────────────────────────────
    n_samples = df["sample_id"].nunique()
    print("test_with_scenario.csv 저장 완료")
    print(f"   경로     : {display_path(dst)}")
    print(f"   전체 행  : {len(df):,}")
    print(f"   전체 샘플: {n_samples}")
    print(f"   컬럼     : {list(df.columns)}")
    print()
    print("   시나리오별 샘플 수:")
    for name, sid in SCENARIO_ID_MAP.items():
        count = df.loc[df["scenario"] == name, "sample_id"].nunique()
        print(f"     [{sid}] {SCENARIO_KR[sid]:<22s} ({name}): {count} 샘플")


if __name__ == "__main__":
    main()
