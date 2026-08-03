from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_json, load_yaml

REQUIRED_CONFIG_PATHS = [
    ("experiment", "id"),
    ("experiment", "seed"),
    ("paths", "data"),
    ("paths", "reference"),
    ("model", "n_embd"),
    ("model", "n_head"),
    ("model", "n_layer"),
    ("training", "lr"),
    ("sampling", "n_sample"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--allow-missing-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = ROOT / "experiments" / args.experiment_id
    config_path = experiment_dir / "config.yaml"
    hypothesis_path = experiment_dir / "hypothesis.json"
    if not config_path.exists() or not hypothesis_path.exists():
        raise SystemExit("Experiment is missing config.yaml or hypothesis.json")

    config = load_yaml(config_path)
    hypothesis = load_json(hypothesis_path)
    for path in REQUIRED_CONFIG_PATHS:
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                raise SystemExit(f"Missing config key: {'.'.join(path)}")
            current = current[key]

    if config["experiment"]["id"] != args.experiment_id:
        raise SystemExit("Config experiment ID does not match directory")
    if hypothesis["id"] != args.experiment_id:
        raise SystemExit("Hypothesis ID does not match directory")
    if int(config["model"]["n_embd"]) % int(config["model"]["n_head"]) != 0:
        raise SystemExit("model.n_embd must be divisible by model.n_head")
    if not 0.0 < float(config["training"]["val_fraction"]) < 1.0:
        raise SystemExit("training.val_fraction must be between 0 and 1")
    if not 0.0 < float(config["sampling"]["top_p"]) <= 1.0:
        raise SystemExit("sampling.top_p must be in (0, 1]")
    if float(config["sampling"]["temperature"]) <= 0:
        raise SystemExit("sampling.temperature must be positive")

    if not args.allow_missing_data:
        for name in ("data", "reference"):
            path = Path(config["paths"][name])
            if not path.exists():
                raise SystemExit(f"Configured {name} file does not exist: {path}")
    print(f"{args.experiment_id} is valid")


if __name__ == "__main__":
    main()
