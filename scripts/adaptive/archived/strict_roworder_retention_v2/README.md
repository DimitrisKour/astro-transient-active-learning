# Strict Row-Order Retention v2 — Archived Reference

Original experiment:
astrominn_entropy135_strict_roworder_retention_v2

Reference trajectory:
Task 4, entropy, seed 135.

Original controller:
strict_roworder_retention.py

Original Slurm worker:
strict_roworder_worker.sbatch

This is an archived implementation, not the entry point
for the new balanced-exploration seed-357 experiments.

The original experiment:
- Reuses the first rejected historical checkpoint.
- Attempts up to four independently permuted row orders.
- Keeps training and loader seeds fixed.
- Requires strict improvement in macro-F1 and balanced accuracy.
- Stops when all retries fail.

Its frozen source tree and runtime artifacts were maintained
separately on Delta. The new experiments must use independent
output directories and must not modify the historical results.
