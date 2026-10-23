# AppleCiDEr representation experiments 5B/5C — isolated pilot v1

**Scope:** The original `Task 5A` evaluated frozen expert and ViT CLS embeddings. This package makes **5B (fixed-manifest)** and **5C (closed-loop)** *ViT plasticity pilots* available as reproducible, explicit-opt-in experiments. It **does not yet implement individual expert-tower fine-tuning**, which needs a revised save/load checkpoint contract. The unrelated newer **Task 5 / 4b** compares L0 baseline restarts and is not changed here.

## Precisely what is being compared

* `vit_full`: ordinary ViT training (all ViT parameters trainable), three pretrained experts frozen. Existing loss-selected ViT checkpoint serialization.
* `vit_head_only`: initialize the *same full ViT* weights, freeze ViT encoder/CLS representation, train only `vit_tower.fc_out`; three pretrained experts remain frozen. The encoder runs in eval mode, so its dropout/BN state does not drift during training. Save a complete ViT state dict so the standard inference/evaluation loader needs no changes.
* Neither arm changes the published experts, data pipeline, class-weight source, augmentation, per-sample loss weighting, validation/test definitions, epoch limit, LR, or optimizer type.

**Critical upstream finding:** `AstroMiNNViTTrainer._freeze_expert_towers()` resets `requires_grad=True` on *all* ViT weights, so merely passing frozen parameters via a `model_factory` is NOT sufficient. During prepare only, this package builds a **frozen source snapshot** that guards and patches `_freeze_expert_towers()` and `_set_training_mode()` to honor an explicit head-only marker. **No checked-in `training.py`, Task 4, or Task 4b source or outputs are modified.** If the upstream function structure changed, freeze fails rather than silently running the wrong experiment.

### 5B — Fixed source acquisitions

Source acquisitions MUST already be completed in either Task 4 or Task 4b (or an audited equivalent with the same layout). 5B reads existing `cycle_summary.json -> cumulative_manifest` and preserves SHA-256 of every manifest, summary and original seed-specific L0 checkpoint.

For each requested (policy, seed, cycle), it trains **both adaptation modes on the identical cumulative training manifest** with the same seed and the SAME original L0 checkpoint as initialization. This deliberate `restart_from_baseline` normalization isolates trainable-layer choice rather than mixing in arbitrary past weights. It produces validation-only object-level accuracy, macro-F1, balanced accuracy, log loss, fit time and trainable parameter counts. **No acquisition, hidden-pool label access, or test-based selection** takes place in 5B.

Recommended first 5B pilot: seeds `135 246 357`, policy `entropy`, source cycles `1 3 5`, modes `vit_full` and `vit_head_only`: **18 new fits (9 paired comparisons)**. Add `entropy_balanced_explore` only if the comparison is informative. Wider runs and cycles 7/9 should be post-pilot decisions.

### 5C — Closed-loop experiment

After reviewing 5B and Task 4b, choose *one* initialization mode (`warm_start` or `restart_from_baseline`) and only the best justified adaptation (`vit_head_only` or `vit_full`). The generated 5C runner reuses the Task 4 closed-loop constructor, acquisition policies, original seed-specific completed L0 checkpoints, original excluded-object overlap audit, training budgets, and fixed evaluation manifests.

At every cycle: evaluate own prior inference model, acquire the policy's next unlabeled objects, reveal only selected labels, train on all cumulative L_t with the selected ViT training arm, save the usual full ViT state and report the fixed validation/test metrics. The experiment is separate from all current suites, and all run IDs and result paths include the mode/initialization.

Recommended 5C pilot: `vit_head_only`, `entropy entropy_balanced_explore`, 3 seeds, cycles `1..5`: **30 new cycle fits**. Compare against the already finished Task 4 or 4b full-ViT trajectories *only if all other settings match*; otherwise a new `vit_full` 5C reference run is available. Later extension to 9 cycles uses a fresh suite ID/plan; do not silently extend an already frozen five-cycle suite.

## Installation on Delta (does not schedule jobs)

From Windows PowerShell:

```powershell
scp "$HOME\Downloads\representation_5bc_pilot_v1.zip" dkourtesis@dt-login.delta.ncsa.illinois.edu:/projects/bcrv/dkourtesis/
```

From Delta:

```bash
source ~/al_delta_setup.sh
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
unzip -n /projects/bcrv/dkourtesis/representation_5bc_pilot_v1.zip -d .
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/astronomy/test_rep5bc.py -q
python scripts/adaptive/rep5bc_build_runner.py --repo "$PWD"
python -m py_compile scripts/run_representation_5c.py
```

The generated `scripts/run_representation_5c.py` is dedicated, and the builder refuses to overwrite a non-identical existing file. Existing Task 4/4b runners and source snapshots stay untouched. These commands are short and CPU-only; perform substantial validation jobs on compute nodes, not the shared login node.

## Prepare 5B after the selected source suite completes

Example below uses the finished warm-start Task 4 source. Replace `SOURCE_SUITE` with `astrominn_closed_loop_baseline_restart_v1` if Task 4b wins. The `REGISTRY` path is an existing Task 4 routed registry; use the actual suitable routing to persistent `/projects` and heavyweight `/work/hdd` storage.

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
P=/projects/bcrv/dkourtesis/applecider_adaptive
REGISTRY="$P/closed_loop/astrominn_closed_loop_warm_v1/registry.toml"
SOURCE_SUITE=astrominn_closed_loop_warm_v1
B="$P/representation/rep5b_vit_plasticity_v1"

python scripts/adaptive/rep5bc.py prepare-b \
  --work-root "$B" --registry "$REGISTRY" \
  --source-suite "$SOURCE_SUITE" \
  --policies entropy --seeds 135 246 357 --cycles 1 3 5

# Always dry run first — no jobs submitted:
python scripts/adaptive/rep5bc.py submit-b --work-root "$B" --parallel 3
```

When you EXPLICITLY want to run the pilot:

```bash
python scripts/adaptive/rep5bc.py submit-b --work-root "$B" \
  --parallel 3 --walltime 00:40:00 --submit
```

Use a separate work-root for another source suite or revised protocol. Every run freezes code under `$B/source` and refuses source/config/manifest drift. The worker uses Delta GPU A40 and runs normal ViT training; `--parallel 3` caps the array at three simultaneous fits. The walltime is adjustable and must be checked against observed fit durations; exceeding it needs an inspection/recovery plan, not blind duplicate submission.

After completion:

```bash
python scripts/adaptive/rep5bc.py report-b --work-root "$B"
cat "$B/report.csv"
```

## Prepare 5C *without submitting*

Once we decide initialization, use one suite per mode and initialization. Example for warm-start head-only:

```bash
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning
P=/projects/bcrv/dkourtesis/applecider_adaptive
REGISTRY="$P/closed_loop/astrominn_closed_loop_warm_v1/registry.toml"
C="$P/representation/rep5c_head_warm_pilot_v1"
S=astrominn_rep5c_vit_head_only_warm_start_v1

python scripts/adaptive/rep5bc.py prepare-c \
  --work-root "$C" --registry "$REGISTRY" \
  --mode vit_head_only --initialization warm_start \
  --suite-id "$S" --policies entropy entropy_balanced_explore \
  --seeds 135 246 357 --cycles 5

python scripts/adaptive/rep5bc.py submit-c --work-root "$C"
```

For `restart_from_baseline` replace both the mode argument and suite ID suffix consistently, e.g.:

```
--mode vit_head_only --initialization restart_from_baseline
--suite-id astrominn_rep5c_vit_head_only_restart_from_baseline_v1
```

Only after the chosen initialization and 5B are reviewed, explicitly launch:

```bash
python scripts/adaptive/rep5bc.py submit-c --work-root "$C" \
  --walltime 00:40:00 --submit
```

This queues 30 jobs in held, serial `afterok` chains (one chain per model seed, at most three independent seed chains). It records job IDs in `$C/submitted.json`. A failed cycle blocks successors. Submission failures cancel newly queued jobs. Do **not** remove `submitted.json` to resubmit blindly. Use standard SLURM commands to inspect failures and then plan controlled recovery of the existing frozen run.

Inspect partial/final results:

```bash
python scripts/adaptive/rep5bc.py report-c --work-root "$C"
cat "$C/report.csv"
```

The reporter audits, for each completed cycle, the inference source, training initialization source, actual training-summary initial checkpoint, budget (473; final c9 remainder 469), non-overlapping selected object IDs and existence of evaluation data; it reports fixed-validation metrics.

## Scientific limitations / gates

1. This first package does **not** implement fine-tuning `lc_tower`, `spatial_tower` or `coord_tower`. That would change the model checkpoint and both prediction/evaluation loading; do not present head-only training as an expert-unfreezing experiment.
2. A head-only model trains only the final classifier; it cannot improve the CLS embedding itself. It is a controlled **fixed-versus-plastic ViT representation** comparison.
3. 5B independently restarts each fit from the original L0, which is intentional. It does not test warm-start-dependent fine-tuning. 5C evaluates the chosen closed-loop initialization.
4. The bootstrap expert-tower pretraining provenance remains an unresolved limitation; the same frozen published towers are reused across arms.
5. 5C comparisons use **object-level validation macro-F1 at equal label budgets**, variance across seeds, balanced accuracy/per-class performance, and compute cost. Test scores are descriptive only; never tune the method on test performance.
6. This package passed CPU-only local tests but has not been run end-to-end on Delta or with private historical data in this environment. Run the repo tests, generated-runner preflight, and a **single representative 5B GPU fit** before submitting full arrays. The test suite can catch source mismatches but cannot prove scientific correctness of real datasets or model loaders.
7. Prepared experiment definitions are intentionally flexible about source suite and initialization because the queued Task 4b results have not been analyzed. Do not select these based on test metrics.
