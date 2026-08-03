# Agentic Materials AutoLab Protocol

## Objective

Improve unconditional SMILES generation through reproducible, hypothesis-driven experiments while preserving chemical validity, novelty, uniqueness, and similarity to the held-out reference property distribution.

## Parallel research model

The AutoLab may test multiple independent hypotheses in parallel. Every parallel worker owns one complete experiment and writes only inside its assigned `experiments/H###/` directory.

All experiments in one comparison batch must:

1. Start from the same frozen baseline commit and configuration.
2. Change exactly one independent variable unless an interaction study is explicitly approved.
3. Use the same dataset split, seed policy, sampling count, descriptors, and evaluation code.
4. Run through Slurm rather than directly on the login node.
5. Produce the standard `results.json` contract.

## Agent responsibilities

### Research manager

Creates a pool of distinct, falsifiable hypotheses and prioritizes experiments with high expected information value. It does not edit code or submit jobs.

### Hypothesis worker

Owns one hypothesis from preregistration through experiment-specific interpretation. It may create a config, request submission, inspect its outputs, and draft a conclusion. It may not alter shared state or promote a baseline.

### Scheduler

Validates experiment isolation, enforces compute limits, and submits Slurm jobs. It does not interpret scientific results.

### Reviewer

Independently reviews completed evidence and assigns `SUPPORTED`, `REJECTED`, `INCONCLUSIVE`, or `INVALID`. It does not propose the next hypothesis.

### Knowledge manager

Updates `research/state.json` and append-only decision records after independent review. It does not modify experiment evidence.

## Frozen components

Autonomous agents must not modify:

- Reference datasets or data splits
- `evaluation/evaluate_generated.py`
- Descriptor definitions
- Statistical significance threshold or correction method
- Completed experiment directories
- Reviewed evidence
- The current baseline without human approval

## Version 1 editing policy

Agents may modify only experiment-specific YAML configuration files and metadata. They must not edit model source code. Architecture-source experiments require a later Git-worktree workflow and explicit human approval.

## Statistical policy

Every experiment reports:

- Validation loss and perplexity
- Validity, uniqueness, and novelty
- Descriptor-level Mann–Whitney U tests
- Multiple-testing-adjusted p-values
- Rank-biserial effect sizes
- Bootstrap confidence intervals for mean differences
- Distributional conformity proxy
- Descriptor distribution plots

A large p-value does not prove that two distributions are identical. Conclusions must use p-values together with effect sizes, confidence intervals, plots, and practical thresholds.

## Compute policy

- Maximum active AutoLab jobs: 3 unless the user explicitly changes the budget.
- Maximum experiments per autonomous batch: 5.
- No GPU training on the login node.
- Stop a batch after three consecutive invalid or infrastructure-failed experiments.
- Do not automatically resubmit failed jobs without diagnosing the failure.

## Git policy

Commit code, configurations, hypotheses, reviews, and compact metrics. Do not commit datasets, checkpoints, WandB run directories, or large logs. Never force-push, rewrite history, or delete prior experiments.

## Required research cycle

1. Read the shared state and prior reviewed experiments.
2. Propose distinct hypotheses that do not duplicate prior work.
3. Preregister one variable, mechanism, primary outcome, and decision criteria per hypothesis.
4. Create isolated experiment directories.
5. Validate all experiments.
6. Submit within the parallel compute budget.
7. Let Slurm train, generate, and evaluate.
8. Review each completed experiment independently.
9. Update shared research state only after review.
10. Recommend the next batch based on accumulated evidence and unresolved uncertainty.
