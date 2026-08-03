# First operating sequence

1. Copy or link `chebi_material_enriched.jsonl` to the path in `configs/baseline.yaml`.
2. Commit the repository before creating experiments.
3. Create and run a `baseline` experiment using the unchanged baseline configuration.
4. Confirm `results.json`, `descriptor_tests.csv`, plots, and the `DONE` marker are produced.
5. Create three configuration-only ablations from the same baseline.
6. Use `scheduler.py` to submit the three jobs in parallel.
7. Review them independently and update shared state.
8. Only after this manual batch succeeds should Claude subagents be permitted to create and submit a batch.

The supplied distributional conformity value is explicitly a proxy based on standardized descriptor mean distance. Replace it with the exact thesis/paper implementation after validating that implementation with the research mentor. Keep the metric name and version recorded so results are not mixed across definitions.
