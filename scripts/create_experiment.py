from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import append_jsonl, git_commit, load_yaml, set_nested, utc_now, write_json, write_yaml

ID_PATTERN = re.compile(r"^H\d{3,}$")


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one isolated hypothesis experiment")
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--parameter", required=True, help="Dotted config key, e.g. training.lr")
    parser.add_argument("--value", required=True, help="New JSON-compatible value")
    parser.add_argument("--baseline", type=Path, default=ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mechanism", default="")
    parser.add_argument("--primary-outcome", default="evaluation.distribution.conformity_proxy")
    parser.add_argument("--support-criterion", action="append", default=[])
    parser.add_argument("--rejection-criterion", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not ID_PATTERN.match(args.id):
        raise SystemExit("Experiment ID must look like H001, H002, ...")

    experiment_dir = ROOT / "experiments" / args.id
    if experiment_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment: {experiment_dir}")

    config = load_yaml(args.baseline)
    proposed_value = parse_scalar(args.value)
    baseline_value = set_nested(config, args.parameter, proposed_value)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    commit = git_commit(ROOT)
    config["experiment"].update(
        {
            "id": args.id,
            "title": args.title,
            "baseline_id": config["experiment"].get("id", "baseline"),
            "baseline_commit": commit,
            "hypothesis": args.hypothesis,
            "independent_variable": {
                "name": args.parameter,
                "baseline_value": baseline_value,
                "proposed_value": proposed_value,
            },
        }
    )

    experiment_dir.mkdir(parents=True)
    (experiment_dir / "outputs").mkdir()
    write_yaml(experiment_dir / "config.yaml", config)

    hypothesis_record = {
        "schema_version": 1,
        "id": args.id,
        "title": args.title,
        "created_at": utc_now(),
        "status": "CREATED",
        "hypothesis": args.hypothesis,
        "mechanism": args.mechanism,
        "baseline_config": str(args.baseline.relative_to(ROOT)),
        "baseline_commit": commit,
        "independent_variable": config["experiment"]["independent_variable"],
        "primary_outcome": args.primary_outcome,
        "support_criteria": args.support_criterion,
        "rejection_criteria": args.rejection_criterion,
    }
    write_json(experiment_dir / "hypothesis.json", hypothesis_record)
    write_json(
        experiment_dir / "status.json",
        {
            "experiment_id": args.id,
            "state": "CREATED",
            "created_at": utc_now(),
            "slurm_job_id": None,
            "updated_at": utc_now(),
        },
    )
    append_jsonl(ROOT / "research" / "hypotheses.jsonl", hypothesis_record)
    print(f"Created {experiment_dir}")
    print(f"Changed only {args.parameter}: {baseline_value!r} -> {proposed_value!r}")


if __name__ == "__main__":
    main()
