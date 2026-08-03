from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import git_commit, load_yaml, utc_now, write_json, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the immutable baseline experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = ROOT / "experiments" / "baseline"
    if experiment_dir.exists():
        if not args.force:
            raise SystemExit("Baseline already exists. Use --force only before any dependent experiment exists.")
        shutil.rmtree(experiment_dir)

    config = load_yaml(args.config)
    config["experiment"].update(
        {
            "id": "baseline",
            "baseline_id": None,
            "baseline_commit": git_commit(ROOT),
            "hypothesis": "Reference run for all controlled comparisons.",
            "independent_variable": {
                "name": None,
                "baseline_value": None,
                "proposed_value": None,
            },
        }
    )
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "outputs").mkdir()
    write_yaml(experiment_dir / "config.yaml", config)
    write_json(
        experiment_dir / "hypothesis.json",
        {
            "schema_version": 1,
            "id": "baseline",
            "title": "Baseline SMILES GPT",
            "created_at": utc_now(),
            "status": "CREATED",
            "hypothesis": "Reference run for all controlled comparisons.",
            "mechanism": "Not applicable",
            "baseline_config": str(args.config.relative_to(ROOT)),
            "baseline_commit": config["experiment"]["baseline_commit"],
            "independent_variable": config["experiment"]["independent_variable"],
            "primary_outcome": "evaluation.distribution.conformity_proxy",
            "support_criteria": [],
            "rejection_criteria": [],
        },
    )
    write_json(
        experiment_dir / "status.json",
        {
            "experiment_id": "baseline",
            "state": "CREATED",
            "created_at": utc_now(),
            "slurm_job_id": None,
            "updated_at": utc_now(),
        },
    )
    print(experiment_dir)


if __name__ == "__main__":
    main()
