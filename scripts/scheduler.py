from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_json, utc_now, write_json


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def parse_job_id(stdout: str) -> str:
    tokens = stdout.strip().split()
    if not tokens or not tokens[-1].isdigit():
        raise RuntimeError(f"Could not parse Slurm job ID from: {stdout!r}")
    return tokens[-1]


def submit(experiment_ids: list[str], max_parallel: int, dry_run: bool) -> None:
    if len(experiment_ids) > max_parallel:
        raise SystemExit(
            f"Requested {len(experiment_ids)} jobs but --max-parallel is {max_parallel}. "
            "Submit a smaller batch or raise the explicit limit."
        )
    if not dry_run and shutil.which("sbatch") is None:
        raise SystemExit("sbatch was not found. Run this command on the HPC login node or use --dry-run.")

    submitted: list[dict[str, str]] = []
    for experiment_id in experiment_ids:
        experiment_dir = ROOT / "experiments" / experiment_id
        if not experiment_dir.exists():
            raise SystemExit(f"Unknown experiment: {experiment_id}")
        status_path = experiment_dir / "status.json"
        status = load_json(status_path)
        if status["state"] not in {"CREATED", "FAILED_SUBMISSION"}:
            raise SystemExit(f"{experiment_id} is already in state {status['state']}")

        validate_command = [sys.executable, "scripts/validate_experiment.py", experiment_id]
        if dry_run:
            validate_command.append("--allow-missing-data")
        run(validate_command)
        render = run([sys.executable, "scripts/render_slurm.py", experiment_id])
        script_path = render.stdout.strip().splitlines()[-1]
        if dry_run:
            job_id = f"DRYRUN-{experiment_id}"
            command = f"sbatch {script_path}"
            print(command)
        else:
            result = run(["sbatch", script_path])
            job_id = parse_job_id(result.stdout)
            print(result.stdout.strip())

        status.update(
            {
                "state": "SUBMITTED" if not dry_run else "DRY_RUN",
                "slurm_job_id": job_id,
                "submitted_at": utc_now(),
                "updated_at": utc_now(),
            }
        )
        write_json(status_path, status)
        submitted.append({"experiment_id": experiment_id, "job_id": job_id})

    print(json.dumps(submitted, indent=2))


def status() -> None:
    records: list[dict[str, str | None]] = []
    for status_path in sorted((ROOT / "experiments").glob("H*/status.json")):
        item = load_json(status_path)
        records.append(
            {
                "experiment_id": item.get("experiment_id"),
                "state": item.get("state"),
                "job_id": item.get("slurm_job_id"),
            }
        )
    print(json.dumps(records, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("experiment_ids", nargs="+")
    submit_parser.add_argument("--max-parallel", type=int, default=3)
    submit_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "submit":
        submit(args.experiment_ids, args.max_parallel, args.dry_run)
    else:
        status()


if __name__ == "__main__":
    main()
