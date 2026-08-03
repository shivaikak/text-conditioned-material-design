#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 EXPERIMENT_ID" >&2
  exit 2
fi

EXPERIMENT_ID="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="$ROOT/experiments/$EXPERIMENT_ID"
CONFIG="$EXPERIMENT_DIR/config.yaml"
OUTPUT_DIR="$EXPERIMENT_DIR/outputs"
PYTHON="${AUTOLAB_PYTHON:-/share/apps/anaconda3/2025.06/bin/python3.13}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$EXPERIMENT_DIR/DONE" "$EXPERIMENT_DIR/FAILED"

update_status() {
  local state="$1"
  local message="${2:-}"
  "$PYTHON" - "$EXPERIMENT_DIR/status.json" "$state" "$message" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
state = sys.argv[2]
message = sys.argv[3]
data = json.loads(path.read_text()) if path.exists() else {}
data.update({
    "state": state,
    "message": message,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", data.get("slurm_job_id")),
})
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data, indent=2) + "\n")
os.replace(tmp, path)
PY
}

on_error() {
  local code=$?
  update_status "FAILED" "Pipeline exited with code $code"
  touch "$EXPERIMENT_DIR/FAILED"
  exit "$code"
}
trap on_error ERR

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

update_status "RUNNING" "Training and evaluation started"

echo "Experiment: $EXPERIMENT_ID"
echo "Started: $(date -Is)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Python: $($PYTHON --version)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" "$ROOT/src/train_generate.py" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON" "$ROOT/evaluation/evaluate_generated.py" \
  --config "$CONFIG" \
  --generated "$OUTPUT_DIR/generated_all_smiles.txt" \
  --output-dir "$OUTPUT_DIR"

"$PYTHON" - "$EXPERIMENT_DIR" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

experiment_dir = Path(sys.argv[1])
outputs = experiment_dir / "outputs"
config_text = (experiment_dir / "config.yaml").read_text()
hypothesis = json.loads((experiment_dir / "hypothesis.json").read_text())
training = json.loads((outputs / "training_summary.json").read_text())
evaluation = json.loads((outputs / "evaluation.json").read_text())
result = {
    "schema_version": 1,
    "experiment_id": hypothesis["id"],
    "status": "COMPLETED",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "hypothesis": hypothesis,
    "training": training,
    "evaluation": evaluation,
    "artifacts": {
        "config": "config.yaml",
        "training_summary": "outputs/training_summary.json",
        "evaluation": "outputs/evaluation.json",
        "descriptor_tests": "outputs/descriptor_tests.csv",
        "generated_smiles": "outputs/generated_all_smiles.txt",
        "plots": "outputs/plots/"
    }
}
tmp = experiment_dir / "results.json.tmp"
tmp.write_text(json.dumps(result, indent=2) + "\n")
os.replace(tmp, experiment_dir / "results.json")
PY

touch "$EXPERIMENT_DIR/DONE"
update_status "COMPLETED" "Training, generation, and evaluation completed"
trap - ERR

echo "Finished: $(date -Is)"
