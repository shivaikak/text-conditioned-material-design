from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    rows = []
    for result_path in sorted((ROOT / "experiments").glob("*/results.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        try:
            score = float(result["evaluation"]["distribution"]["conformity_proxy"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"experiment": result["experiment_id"], "conformity": score})
    if not rows:
        raise SystemExit("No completed results were found")
    frame = pd.DataFrame(rows)
    frame["best_so_far"] = frame["conformity"].cummax()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    frame.to_csv(reports / "progress.csv", index=False)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.step(frame["experiment"], frame["best_so_far"], where="post", label="Best so far")
    axis.scatter(frame["experiment"], frame["conformity"], alpha=0.65, label="Each experiment")
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Distributional conformity proxy")
    axis.set_title("AutoLab research progress")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    figure.tight_layout()
    figure.savefig(reports / "best_so_far_step.png", dpi=180)
    print(reports / "best_so_far_step.png")


if __name__ == "__main__":
    main()
