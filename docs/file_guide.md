# File guide

## Root

- `README.md`: setup and operating commands.
- `CLAUDE.md`: permanent scientific, safety, parallelism, Git, and HPC rules for Claude Code.
- `requirements.txt`: Python dependencies matching the uploaded environment.
- `.gitignore`: prevents datasets, checkpoints, WandB runs, and large experiment outputs from entering Git.

## Model and evaluation

- `src/common.py`: safe YAML and JSON I/O, Git commit lookup, nested configuration edits, hashes, and timestamps.
- `src/train_generate.py`: configuration-driven version of the uploaded GPT-style SMILES model. It trains through Lightning, saves an isolated checkpoint, generates samples, and writes `training_summary.json`.
- `evaluation/evaluate_generated.py`: computes RDKit descriptors, validity, uniqueness, novelty, Mann–Whitney U tests, FDR-adjusted p-values, rank-biserial effect sizes, bootstrap confidence intervals, plots, and a documented conformity proxy.

## Experiment control

- `scripts/init_baseline.py`: creates `experiments/baseline/` from `configs/baseline.yaml`.
- `scripts/create_experiment.py`: preregisters one hypothesis and changes exactly one dotted YAML parameter.
- `scripts/create_batch.py`: turns research-manager JSON into several isolated experiment folders.
- `scripts/validate_experiment.py`: checks config integrity and optional HPC data paths.
- `scripts/render_slurm.py`: creates an experiment-specific `submit.slurm` using the resource policy in its config.
- `scripts/scheduler.py`: validates and submits one or more independent jobs with an explicit parallel limit.
- `scripts/review_experiment.py`: compares a completed run with the baseline using practical preservation thresholds.
- `scripts/update_research_state.py`: updates shared state only after review.
- `scripts/update_step_plot.py`: creates the best-so-far staircase plot from completed results.
- `scripts/validate_setup.py`: confirms the repository scaffold and reports whether Slurm is available.

## HPC

- `slurm/experiment.slurm`: runs the full atomic pipeline: train, generate, evaluate, consolidate evidence, and create `DONE` or `FAILED` markers.

## Agent definitions

- `.claude/agents/research-manager.md`: proposes a diverse parallel hypothesis batch.
- `.claude/agents/critic.md`: checks hypotheses before GPU spending.
- `.claude/agents/hypothesis-worker.md`: owns one isolated hypothesis cycle.
- `.claude/agents/scheduler.md`: handles Slurm operations without scientific interpretation.
- `.claude/agents/reviewer.md`: independently evaluates evidence.
- `.claude/agents/knowledge-manager.md`: updates shared research memory after review.

## Research memory

- `research/state.json`: current baseline, active work, and reviewed decisions.
- `research/hypotheses.jsonl`: append-only hypothesis history.
- `research/batches.jsonl`: append-only batch membership records.

## Legacy

The unchanged uploaded training, analysis, Slurm, and Claude files are preserved in `legacy/` for provenance and comparison.
