from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import append_jsonl, load_json, utc_now, write_json


def unique_append(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def remove_if_present(items: list[str], value: str) -> None:
    if value in items:
        items.remove(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    args = parser.parse_args()

    experiment_dir = ROOT / "experiments" / args.experiment_id
    review = load_json(experiment_dir / "review.json")
    state_path = ROOT / "research" / "state.json"
    state = load_json(state_path)
    decision = review["decision"]

    for key in (
        "running_experiments",
        "completed_experiments",
        "reviewed_experiments",
        "failed_experiments",
        "supported_hypotheses",
        "rejected_hypotheses",
        "inconclusive_hypotheses",
    ):
        state.setdefault(key, [])

    remove_if_present(state["running_experiments"], args.experiment_id)
    unique_append(state["completed_experiments"], args.experiment_id)
    unique_append(state["reviewed_experiments"], args.experiment_id)
    for key in ("supported_hypotheses", "rejected_hypotheses", "inconclusive_hypotheses"):
        remove_if_present(state[key], args.experiment_id)

    if decision == "SUPPORTED":
        unique_append(state["supported_hypotheses"], args.experiment_id)
    elif decision == "REJECTED":
        unique_append(state["rejected_hypotheses"], args.experiment_id)
    elif decision == "INCONCLUSIVE":
        unique_append(state["inconclusive_hypotheses"], args.experiment_id)
    elif decision == "INVALID":
        unique_append(state["failed_experiments"], args.experiment_id)

    state["updated_at"] = utc_now()
    write_json(state_path, state)
    append_jsonl(
        ROOT / "research" / "decisions.jsonl",
        {
            "experiment_id": args.experiment_id,
            "decision": decision,
            "recorded_at": utc_now(),
            "review_path": str((experiment_dir / "review.json").relative_to(ROOT)),
        },
    )
    print(f"Recorded {args.experiment_id} as {decision}")


if __name__ == "__main__":
    main()
