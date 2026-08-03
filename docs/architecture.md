# AutoLab architecture

The system separates parallel scientific work from shared governance.

```text
Research Manager
      |
      v
Candidate hypothesis pool
      |
      v
Critic and Scheduler
      |
      +----------------+----------------+----------------+
      v                v                v                v
Hypothesis H001   Hypothesis H002   Hypothesis H003   queued work
      |                |                |
      v                v                v
Slurm job          Slurm job          Slurm job
      |                |                |
      v                v                v
Train, generate, evaluate, statistical tests
      |                |                |
      +----------------+----------------+
                       v
              Independent Reviewer
                       |
                       v
               Knowledge Manager
```

Configuration-only experiments share committed source code but use separate experiment directories and Slurm jobs. Future source-code architecture experiments must use one Git worktree per hypothesis.
