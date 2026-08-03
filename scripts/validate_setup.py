from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_yaml


def main() -> None:
    required_files = [
        ROOT / "configs" / "baseline.yaml",
        ROOT / "src" / "train_generate.py",
        ROOT / "evaluation" / "evaluate_generated.py",
        ROOT / "slurm" / "experiment.slurm",
        ROOT / "research" / "state.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))
    config = load_yaml(ROOT / "configs" / "baseline.yaml")
    print("Repository structure: OK")
    print(f"Configured Python: {config['slurm']['python']}")
    print(f"Dataset path: {config['paths']['data']}")
    print(f"sbatch available: {'yes' if shutil.which('sbatch') else 'no (expected off HPC)'}")
    print("Run scripts/validate_experiment.py with --allow-missing-data for local config checks.")


if __name__ == "__main__":
    main()
