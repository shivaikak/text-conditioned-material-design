# Agentic Materials AutoLab

A parallel, hypothesis-driven research framework for improving unconditional molecular generation models on an HPC cluster.

The repository adapts an existing character-level SMILES Transformer workflow into isolated experiments that can be created, submitted through Slurm, evaluated consistently, reviewed independently, and recorded as structured research evidence.

## Core workflow

1. A research agent proposes several distinct, falsifiable hypotheses.
2. Each hypothesis becomes an immutable experiment directory with its own configuration.
3. The scheduler submits multiple independent Slurm jobs, subject to a parallel-job limit.
4. Each job trains a model, generates molecules, evaluates chemical and statistical properties, and writes machine-readable results.
5. A reviewer classifies each hypothesis as `SUPPORTED`, `REJECTED`, `INCONCLUSIVE`, or `INVALID`.
6. A knowledge manager updates the shared research state only after review.

## Repository layout

```text
src/                  Model training and generation code
configs/              Baseline configuration
scripts/              Experiment creation, scheduling, review, and reporting
slurm/                 Slurm entry point for one complete experiment
evaluation/            Generated-versus-reference statistical analysis
research/              Shared AutoLab state and append-only research records
experiments/           One immutable directory per hypothesis
.claude/agents/        Claude Code subagent definitions
docs/                  Architecture and operating instructions
legacy/                Original uploaded scripts, preserved unchanged
```

## Quick start on NYU Torch

```bash
cd /scratch/sk10731
git clone <YOUR_PRIVATE_REPOSITORY_URL> agentic-materials-autolab
cd agentic-materials-autolab

python scripts/validate_setup.py

python scripts/create_experiment.py \
  --id H001 \
  --title "Lower learning rate" \
  --hypothesis "Reducing the learning rate from 3e-4 to 2e-4 improves held-out distributional conformity without reducing validity." \
  --parameter training.lr \
  --value 0.0002

python scripts/scheduler.py submit H001
```

Check the job:

```bash
python scripts/scheduler.py status
squeue -u "$USER"
```

After completion:

```bash
python scripts/review_experiment.py H001
python scripts/update_research_state.py H001
python scripts/update_step_plot.py
```

## Parallel batch example

Create three hypotheses from the same frozen baseline, then submit them together:

```bash
python scripts/scheduler.py submit H001 H002 H003 --max-parallel 3
```

For fair ablations, all experiments in one batch should use the same baseline commit and change only one independent variable each.

## Important safety rules

- Never train directly on the login node.
- Never let parallel workers edit the same source files or experiment directory.
- Keep test data, data splits, descriptor definitions, and evaluation code frozen during an autonomous batch.
- Start with configuration-only experiments. Use Git worktrees for future experiments that require architecture source-code changes.
- Do not promote a new baseline automatically until repeated-seed results and human review are complete.

## Existing model preserved

The original uploaded training, analysis, Slurm, and Claude guidance files are preserved under `legacy/`. The new pipeline retains the same core Transformer architecture and RDKit descriptor set while adding isolated output paths, structured JSON evidence, fixed seeds, statistical tests, and parallel scheduling.
