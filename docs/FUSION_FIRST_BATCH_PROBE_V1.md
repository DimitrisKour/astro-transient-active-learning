Read-only-to-experiment first-batch diagnostic for the full historical fast-cache trainer.

This modifies model weights ONLY in memory and saves no model checkpoint. No scientific
training runs, oracle operations, or test-set data are accessed. The original failed
run/output and frozen experiment plan remain unmodified.

Install into active-learning root with unzip -o; new files only.
Compile/check shell: python -m py_compile scripts/adaptive/fusion_first_batch_probe.py; bash -n scripts/adaptive/fusion_first_batch_probe.sbatch
Submit using sbatch --output=... --error=... scripts/adaptive/fusion_first_batch_probe.sbatch
Log prints start/end for each batch fetch, transfer, forward, backward so timeout
still identifies the blocking phase. Writes partial.json continuously and probe.json
on full completion under /projects/bcrv/dkourtesis/applecider_adaptive/new_image_first_batch_probe_v1.
