from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = ROOT / "experiments" / args.experiment_id
    config = load_yaml(experiment_dir / "config.yaml")
    slurm = config["slurm"]
    script = experiment_dir / "submit.slurm"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.experiment_id}",
        f"#SBATCH --account={slurm['account']}",
        f"#SBATCH --partition={slurm['partition']}",
        f"#SBATCH --gres={slurm['gres']}",
        f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}",
        f"#SBATCH --mem={slurm['mem']}",
        f"#SBATCH --time={slurm['time']}",
        f"#SBATCH --output={experiment_dir}/slurm-%j.out",
        f"#SBATCH --error={experiment_dir}/slurm-%j.err",
        f"#SBATCH --mail-user={slurm['mail_user']}",
        f"#SBATCH --mail-type={slurm['mail_type']}",
        f"#SBATCH --chdir={ROOT}",
        "",
        "set -euo pipefail",
        f"export AUTOLAB_PYTHON={shlex.quote(str(slurm['python']))}",
        f"bash {shlex.quote(str(ROOT / 'slurm' / 'experiment.slurm'))} {shlex.quote(args.experiment_id)}",
        "",
    ]
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o750)
    print(script)


if __name__ == "__main__":
    main()
