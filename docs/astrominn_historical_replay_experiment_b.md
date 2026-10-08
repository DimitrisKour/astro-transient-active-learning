# Experiment B: historical AstroMiNN active-learning replay

Experiment B asks whether uncertainty acquisition improves a model trained
from a deliberately small labelled subset of the original historical
training split. It is independent of Experiment A's post-2025 BOOM/TNS pool.

This guide uses the extracted `al_pipeline` package in the `al-pipeline`
environment. Current Delta dataset, checkpoint, and legacy model-code paths
are retained. Before any launcher command, run this setup in the current
shell, including a new compute-node shell:

```bash
conda activate al-pipeline
cd /projects/bcrv/dkourtesis/applecider-al-workspace/active-learning

al_python() {
  env LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" python "$@"
}
```

See [Delta installation](delta_installation.md) for the runtime workaround,
first-time installation, and suite/resume rules. Use unused suite IDs for
new trials. The full-budget and one-epoch examples deliberately use different
IDs so an execution-check checkpoint cannot become a reproduction checkpoint.

## Fixed data design

The original historical train manifest is partitioned at object level:

- `L0`: a deterministic 10% stratified sample of every AstroMiNN target
  class, with at least five objects from each rare class;
- `U0`: every other historical training object, with its label hidden from
  inference and acquisition;
- validation and test: the original published manifests, unchanged.

Every alert-stage row for an object remains with that object. The candidate
pool therefore contains all historical-compatible stages from `U0`, while
acquisition operates once per object using the latest available stage. When
an object is acquired, all of its stage rows enter cumulative `L_t`.

The split seed is fixed at 135 across all model seeds. This keeps `L0`, `U0`,
validation, and test identical for paired strategy comparisons. Sampling the
same fraction from every class approximately preserves the historical class
distribution. The five-object floor prevents TDE or another rare class from
being absent or represented by only one or two objects.

The model seed controls both the random ViT initialization and loader
randomness (shuffle and training augmentation). All strategies within one
model seed share the exact same bootstrap checkpoint; seeds 135, 246, and
357 are independent training replicas of the same fixed data split.

The builder enforces:

- no `L0/U0`, train/validation, train/test, or validation/test object overlap;
- exactly one target class per object;
- complete support for all five classes in `L0`;
- zero-based alert numbers within the configured `0..10` boundary;
- a label-free candidate manifest;
- deterministic, idempotent manifests and content hashes;
- unchanged validation and test manifest hashes.

No sample arrays are copied. The new manifests point to the immutable
historical `.npy` files, and the preflight verifies that every candidate file
exists.

By default, each acquisition cycle requests another 10% of the complete
historical training-object count, not 10% of the shrinking pool. With eight
cycles, the experiment therefore records the trajectory at approximately
10%, 20%, ..., 90% labelled. The final 10% remains hidden, and every cycle has
the same label budget.

## Training design

Experiment B does not use the published ViT checkpoint. For each model seed:

1. the three pretrained expert towers are loaded and frozen;
2. a randomly initialized ViT tower is trained on `L0` once, producing the
   common cycle-zero bootstrap checkpoint shared by all strategies;
3. cycle 1 acquisitions are scored by that trained `L0` model;
4. after every acquisition, the ViT tower is reinitialized from the same
   seed and retrained on cumulative `L_t`;
5. the resulting model scores the next `U_t` and is evaluated on the fixed
   original validation and test manifests.

This is a true retraining comparison at the ViT level. The expert towers are
not retrained because the recovered historical AstroMiNN procedure treats
them as pretrained feature extractors.

The published checkpoint path in the dataset profile is used only to locate
the neighboring expert-tower artifacts and model configuration. Its ViT
weights are never loaded by this experiment.

Class weights are recomputed from the currently revealed training manifest
only and normalized to mean one. The original fixed class-count reference is
not used because it contains the full historical class distribution and
would leak information from `U0`. Training otherwise retains per-sample
historical weighting: an object with more available early stages contributes
more rows, matching the original AstroMiNN training convention.

## Evaluation protocol

The original historical validation and test manifests are reused without
resampling or adding post-2025 objects. Validation loss controls the learning
rate, early stopping, and best-epoch checkpoint selection. In a full AL run,
the `L0` bootstrap and the checkpoint after every acquisition/retraining cycle are
evaluated on both original splits. A bootstrap-only launch records training
and validation-loss output; baseline/test evaluation is performed when the
AL suite starts. Test metrics are recorded for the final frozen-policy
learning curve but never affect acquisition,
training, early stopping, or checkpoint selection. Policy choices must be
frozen from the pilot and validation evidence before the final test curves
are interpreted.

The post-2025 Experiment A manifest is accepted only through
`--assert-excluded-manifest`. The builder reads its object-ID column to prove
zero overlap with historical train/validation/test and records its hash; it
never uses its labels and never loads its samples or checkpoint into
Experiment B.

## 1. CPU preflight

Run from the repository root in the `al-pipeline` environment with the runtime
helper above. This creates and audits only small CSV/JSON manifests; it does
not train or run inference.

```bash
al_python scripts/run_astrominn_historical_replay.py \
  --registry configs/delta.toml \
  --suite-id al_pipeline_experiment_b_v1 \
  --strategies random margin entropy \
  --initial-fraction 0.10 \
  --minimum-initial-per-class 5 \
  --split-seed 135 \
  --acquisition-fraction 0.10 \
  --max-cycles 8 \
  --max-epochs 50 \
  --patience 10 \
  --batch-size 512 \
  --validation-batch-size 2048 \
  --learning-rate 0.0001 \
  --seed 135 \
  --device cpu \
  --assert-excluded-manifest /projects/bcrv/dkourtesis/applecider_adaptive/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv
```

Review the printed class counts, paths, budget, and policy. Also inspect:

```bash
cat /projects/bcrv/dkourtesis/applecider_adaptive/runs/al_pipeline_experiment_b_v1/shared/historical_replay_split/split_summary.json
```

Require:

- an initial labelled fraction close to 0.10;
- every initial class count close to 10% of its historical training count,
  except where the five-object rare-class floor applies;
- `historical_objects = initial_objects + candidate_objects`;
- all four overlap checks equal zero;
- the external Experiment A overlap audit equals zero;
- a planned final labelled fraction close to 0.90;
- maximum alert 10;
- no missing candidate files;
- non-empty hashes for the source and generated manifests.

For the unchanged current historical source, expect 4,729 training objects,
476 initially labelled objects (2,890 rows), 4,253 candidate objects
(26,271 rows), and 473 acquisitions per cycle. All five target classes must
be represented in `L0`. Eight cycles plan 4,260 labelled objects, approximately
90% of the historical training objects. The exclusion audit covers 463
external objects with zero overlap. These numbers check the current source;
they are not hardcoded requirements for a future dataset.

## 2. Full-budget bootstrap-only GPU stage

After the preflight, this optional first stage trains only the seed-135 `L0`
model and stops before acquisition. It uses the same full-budget policy and
suite ID as the preflight. Run inside a GPU allocation after repeating the
runtime setup:

```bash
al_python -u scripts/run_astrominn_historical_replay.py \
  --registry configs/delta.toml \
  --suite-id al_pipeline_experiment_b_v1 \
  --strategies random margin entropy \
  --initial-fraction 0.10 \
  --minimum-initial-per-class 5 \
  --split-seed 135 \
  --acquisition-fraction 0.10 \
  --max-cycles 8 \
  --max-epochs 50 \
  --patience 10 \
  --batch-size 512 \
  --validation-batch-size 2048 \
  --learning-rate 0.0001 \
  --seed 135 \
  --device cuda \
  --assert-excluded-manifest /projects/bcrv/dkourtesis/applecider_adaptive/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv \
  --run \
  --bootstrap-only
```

Inspect the bootstrap training summary before any AL cycle. Confirm finite
losses, a sensible best epoch, `initialization = random_vit`, and
`class_weighting.source = training_manifest`.

To continue into the full eight-cycle random/margin/entropy comparison,
remove only `--bootstrap-only` and keep `--run`, the suite ID, and every
other argument unchanged. The completed bootstrap is reused by all three
strategies. On interruption, rerun that same command to resume. Do not reuse
the original `astrominn_historical_replay_v1` directory through this guide.

## 3. Separate one-epoch execution check

Use a separate suite ID, entropy acquisition, and one cycle. This is a small
end-to-end check of bootstrap, acquisition, label reveal, cumulative-manifest
construction, retraining from random, and evaluation. Run inside a GPU
allocation after the runtime setup:

```bash
al_python -u scripts/run_astrominn_historical_replay.py \
  --registry configs/delta.toml \
  --suite-id al_pipeline_check_b_v1 \
  --strategies entropy \
  --initial-fraction 0.10 \
  --minimum-initial-per-class 5 \
  --split-seed 135 \
  --acquisition-fraction 0.10 \
  --max-cycles 1 \
  --max-epochs 1 \
  --patience 1 \
  --seed 135 \
  --device cuda \
  --assert-excluded-manifest /projects/bcrv/dkourtesis/applecider_adaptive/runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv \
  --run
```

Successful completion prints `"entropy": [1]`. This trains one bootstrap
epoch and one cycle-training epoch, acquiring 473 objects and reaching
949 labelled objects. The one-epoch metrics check execution only. Keep this
suite separate from full-budget runs; changing its epoch arguments could
reuse its already-completed checkpoints.

Verify that the execution check:

- trains the bootstrap from random on exactly `L0`;
- evaluates cycle zero on the unchanged validation/test sets;
- selects the printed budget of unique `U0` objects without first reading labels;
- produces `L1 = L0 union A1` with every stage for each acquired object;
- retrains the ViT from random on `L1`;
- records manifest-derived class counts and weights;
- leaves the three expert towers frozen;
- writes complete resumable state and evaluation artifacts.

The full-budget example retains the existing 50-epoch maximum and patience-10
policy. Any later policy study or additional model-seed comparison should
use a separate suite and keep split seed 135 fixed. Freeze the scientific
policy before interpreting final test curves.
