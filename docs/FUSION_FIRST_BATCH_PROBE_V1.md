> **Archive notice:** This document reproduces an earlier fusion-image prototype or diagnostic. The scripts below are now under `scripts/archive/fusion_image/`, rather than the supported full-history resident-cache/Task 0/Task 1 workflow. Consult the archive README before any historical submission. Existing frozen snapshots must not be overwritten.

Read-only-to-experiment first-batch diagnostic for the full historical fast-cache trainer.

This modifies model weights ONLY in memory and saves no model checkpoint. No scientific
training runs, oracle operations, or test-set data are accessed. The original failed
run/output and frozen experiment plan remain unmodified.

Install into active-learning root with unzip -o; new files only.
Compile/check shell: python -m py_compile scripts/archive/fusion_image/early_diagnostics/fusion_first_batch_probe.py; bash -n scripts/archive/fusion_image/early_diagnostics/fusion_first_batch_probe.sbatch
Submit using sbatch --output=... --error=... scripts/archive/fusion_image/early_diagnostics/fusion_first_batch_probe.sbatch
Log prints start/end for each batch fetch, transfer, forward, backward so timeout
still identifies the blocking phase. Writes partial.json continuously and probe.json
on full completion under /projects/bcrv/dkourtesis/applecider_adaptive/new_image_first_batch_probe_v1.
