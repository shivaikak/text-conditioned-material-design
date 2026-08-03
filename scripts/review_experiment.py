from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_json, utc_now, write_json


def metric(document: dict[str, Any], dotted_path: str) -> float:
    current: Any = document
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return float(current)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--baseline-id", default="baseline")
    parser.add_argument("--minimum-conformity-gain", type=float, default=0.005)
    parser.add_argument("--maximum-validity-drop", type=float, default=0.01)
    parser.add_argument("--maximum-novelty-drop", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = ROOT / "experiments" / args.experiment_id
    results_path = experiment_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}; the experiment has not completed successfully")
    result = load_json(results_path)
    baseline_path = ROOT / "experiments" / args.baseline_id / "results.json"

    integrity_issues: list[str] = []
    if result.get("status") != "COMPLETED":
        integrity_issues.append(f"Result status is {result.get('status')}")
    if not (experiment_dir / "DONE").exists():
        integrity_issues.append("DONE marker is missing")
    hypothesis = load_json(experiment_dir / "hypothesis.json")
    independent = hypothesis.get("independent_variable", {})
    if not independent.get("name"):
        integrity_issues.append("No independent variable was preregistered")

    review: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "reviewed_at": utc_now(),
        "integrity_issues": integrity_issues,
        "decision": "INVALID" if integrity_issues else "INCONCLUSIVE",
        "reasoning": [],
        "baseline_id": args.baseline_id,
    }

    if not integrity_issues and baseline_path.exists():
        baseline = load_json(baseline_path)
        conformity = metric(result, "evaluation.distribution.conformity_proxy")
        baseline_conformity = metric(baseline, "evaluation.distribution.conformity_proxy")
        validity = metric(result, "evaluation.generation.validity")
        baseline_validity = metric(baseline, "evaluation.generation.validity")
        novelty = metric(result, "evaluation.generation.novelty")
        baseline_novelty = metric(baseline, "evaluation.generation.novelty")
        conformity_gain = conformity - baseline_conformity
        validity_change = validity - baseline_validity
        novelty_change = novelty - baseline_novelty
        review["comparisons"] = {
            "conformity_gain": conformity_gain,
            "validity_change": validity_change,
            "novelty_change": novelty_change,
        }
        if (
            conformity_gain >= args.minimum_conformity_gain
            and validity_change >= -args.maximum_validity_drop
            and novelty_change >= -args.maximum_novelty_drop
        ):
            review["decision"] = "SUPPORTED"
            review["reasoning"].append("Predefined conformity improvement and preservation thresholds were met.")
        elif validity_change < -args.maximum_validity_drop or novelty_change < -args.maximum_novelty_drop:
            review["decision"] = "REJECTED"
            review["reasoning"].append("A protected generation metric declined beyond its allowed threshold.")
        elif conformity_gain < 0:
            review["decision"] = "REJECTED"
            review["reasoning"].append("Distributional conformity was lower than the baseline.")
        else:
            review["decision"] = "INCONCLUSIVE"
            review["reasoning"].append("The effect did not cross the preregistered practical threshold.")
    elif not integrity_issues:
        review["reasoning"].append(
            "No completed baseline results were found. The experiment is valid but cannot yet be classified comparatively."
        )

    review["warning"] = (
        "This automatic review is a protocol check, not a substitute for scientific judgment. "
        "Repeated seeds and a paired cross-seed analysis are required before baseline promotion."
    )
    write_json(experiment_dir / "review.json", review)
    print(f"{args.experiment_id}: {review['decision']}")


if __name__ == "__main__":
    main()
