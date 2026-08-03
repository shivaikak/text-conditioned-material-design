from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a batch from research-manager JSON")
    parser.add_argument("proposal", type=Path)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    batch_id = proposal["batch_id"]
    candidates = proposal["candidates"]
    created: list[str] = []

    for candidate in candidates:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "create_experiment.py"),
            "--id",
            candidate["id"],
            "--title",
            candidate["title"],
            "--hypothesis",
            candidate["hypothesis"],
            "--parameter",
            candidate["parameter"],
            "--value",
            json.dumps(candidate["proposed_value"]),
            "--mechanism",
            candidate.get("mechanism", ""),
            "--primary-outcome",
            candidate.get("primary_outcome", "evaluation.distribution.conformity_proxy"),
        ]
        for criterion in candidate.get("support_criteria", []):
            command.extend(["--support-criterion", criterion])
        for criterion in candidate.get("rejection_criteria", []):
            command.extend(["--rejection-criterion", criterion])
        subprocess.run(command, cwd=ROOT, check=True)
        created.append(candidate["id"])

    with (ROOT / "research" / "batches.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"batch_id": batch_id, "experiments": created}) + "\n")
    print(json.dumps({"batch_id": batch_id, "experiments": created}, indent=2))


if __name__ == "__main__":
    main()
