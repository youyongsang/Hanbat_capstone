"""Experiment result logging utilities.

장예나 Stage 3 담당 파일.
실험 결과를 comparison_summary.csv 형식으로 저장하는 유틸리티 함수 모음.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Union


SUMMARY_COLUMNS = [
    "model",
    "accuracy",
    "avg_inference_ms",
    "exit1_rate",
    "exit2_rate",
    "exit3_rate",
    "false_congestion_rate",
    "unnecessary_switch_rate",
]


def save_results(
    model_name: str,
    accuracy: float,
    inference_time: float,
    exit_rates: Optional[Dict[int, float]],
    save_path: Union[str, Path],
    false_congestion_rate: float = 0.0,
    unnecessary_switch_rate: float = 0.0,
) -> None:
    """실험 결과 한 행을 comparison_summary.csv에 저장(추가).

    파일이 없으면 헤더와 함께 새로 생성하고,
    이미 있으면 헤더 없이 행만 추가(append)한다.

    Args:
        model_name: 모델 식별자 (예: 'baseline_lstm', 'early_exit_fixed').
        accuracy: 전체 테스트 정확도 — 비율 값(0 ~ 1).
        inference_time: 샘플당 평균 추론 시간 (ms).
        exit_rates: {exit_point: rate} dict (키 1, 2, 3) — Early Exit 모델만 해당.
                    Early Exit 없는 모델은 None 또는 {} 전달.
        save_path: 저장할 CSV 파일 경로.
        false_congestion_rate: 정상 구간(label=0)을 혼잡(label 1~3)으로 오판한 비율 (0 ~ 1).
        unnecessary_switch_rate: 정상 구간(label=0)에서 실제 채널 전환이 발생한 비율 (0 ~ 1).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = save_path.exists()
    exit_rates = exit_rates or {}

    row = {
        "model": model_name,
        "accuracy": round(float(accuracy), 4),
        "avg_inference_ms": round(float(inference_time), 4),
        "exit1_rate": round(float(exit_rates[1]), 4) if 1 in exit_rates else "",
        "exit2_rate": round(float(exit_rates[2]), 4) if 2 in exit_rates else "",
        "exit3_rate": round(float(exit_rates[3]), 4) if 3 in exit_rates else "",
        "false_congestion_rate": round(float(false_congestion_rate), 4),
        "unnecessary_switch_rate": round(float(unnecessary_switch_rate), 4),
    }

    with save_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_summary(save_path: Union[str, Path]) -> List[dict]:
    """comparison_summary.csv를 읽어 행 dict 리스트로 반환.

    파일이 없으면 빈 리스트를 반환한다.
    """
    save_path = Path(save_path)
    if not save_path.exists():
        return []
    with save_path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def print_summary(save_path: Union[str, Path]) -> None:
    """comparison_summary.csv 내용을 콘솔에 출력."""
    rows = load_summary(save_path)
    if not rows:
        print(f"[비어 있음] {save_path}")
        return

    header = f"{'모델':<40} {'정확도':>8} {'지연(ms)':>10} {'Exit1':>7} {'Exit2':>7} {'Exit3':>7} {'혼잡오판':>10} {'불필요전환':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<40} "
            f"{r['accuracy']:>8} "
            f"{r['avg_inference_ms']:>10} "
            f"{r['exit1_rate']:>7} "
            f"{r['exit2_rate']:>7} "
            f"{r['exit3_rate']:>7} "
            f"{r.get('false_congestion_rate', ''):>10} "
            f"{r['unnecessary_switch_rate']:>10}"
        )
