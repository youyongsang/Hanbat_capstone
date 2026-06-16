"""Analyze all Raspberry Pi staged ONNX results and create final graphs.

This is a convenience wrapper around:
- analyze_pi_results.py
- visualize_pi_stage_results.py

Run from the repository root:
    python project/scripts/analyze_and_visualize_pi_stage_results.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "hojung"


STAGE_RUNS = [
    ("pi_fixed_staged_fp32_results.csv", "pi_fixed_staged_fp32_analysis"),
    ("pi_fixed_staged_int8_results.csv", "pi_fixed_staged_int8_analysis"),
    ("pi_dynamic_staged_fp32_results.csv", "pi_dynamic_staged_fp32_analysis"),
    ("pi_dynamic_staged_int8_results.csv", "pi_dynamic_staged_int8_analysis"),
]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_command(args: list[str]) -> None:
    print(" ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def main() -> None:
    missing = [RESULTS_DIR / filename for filename, _ in STAGE_RUNS if not (RESULTS_DIR / filename).exists()]
    if missing:
        print("Missing Raspberry Pi staged result files:")
        for path in missing:
            print(f"- {display_path(path)}")
        sys.exit(1)

    analyzer = PROJECT_ROOT / "scripts" / "analyze_pi_results.py"
    visualizer = PROJECT_ROOT / "scripts" / "visualize_pi_stage_results.py"

    for filename, analysis_name in STAGE_RUNS:
        run_command(
            [
                sys.executable,
                str(analyzer),
                "--input",
                str(RESULTS_DIR / filename),
                "--output-dir",
                str(RESULTS_DIR),
                "--name",
                analysis_name,
            ]
        )

    run_command([sys.executable, str(visualizer)])

    print("\nRaspberry Pi staged analysis and figures are ready.")
    print(f"- {display_path(PROJECT_ROOT / 'results' / 'final_figures' / 'pi_stage_comparison.csv')}")
    print(f"- {display_path(PROJECT_ROOT / 'results' / 'final_figures' / 'pi_stage_accuracy_latency.png')}")
    print(f"- {display_path(PROJECT_ROOT / 'results' / 'final_figures' / 'pi_stage_exit_distribution.png')}")


if __name__ == "__main__":
    main()
