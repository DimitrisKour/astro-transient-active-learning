# Architecture and migration

## This extraction

The package has its own `al_pipeline` namespace and packaging metadata. The
old fork and upstream AppleCiDEr no longer own the extracted AL source files.
`docs/source_provenance.json`, `source_map.json`, and `symbol_map.json` record
the extraction source and import changes.

Generic batches contain classifier predictions, selected IDs, and labels.
Alert-aware inference provenance lives in `astronomy/astrominn/types.py`.
Generic evidence and oracle facades do not import BOOM or catalogue rules.
Survey manifests and cumulative acquisition CSVs remain in the astronomy
layer because their `survey`/`alert` columns are part of the current data format.
The generic training/evaluation contracts do not interpret manifest rows.

## Preserved scientific behavior

Acquisition scores, tie handling, object selection, pool transitions,
label-resolution policies, model classes and ordering, preprocessing,
initialization policies, training requests, split creation, manifest hashes,
and interruption/recovery guards retain the source implementation.
The astronomical inference-result type and experiment-policy types were moved
without changing their fields or validation.

Existing absolute Delta paths remain in the astronomy adapter and local
registry. No datasets, checkpoints, manifests, or old run directories are moved.
The first verification uses fresh suite IDs. Reading old results for reporting
is supported by the existing artifact schema; resuming an old live run through
the extracted package is not part of this first verification.

## Retired provider

The unused provider source and its dedicated tests are excluded. Candidate
inspection keeps sample loading, model preprocessing, plots, and caller-supplied
metadata, but no longer reads the retired provider's Parquet transfers.
Use `metadata_fields` to select already-attached diagnostic columns. Shared
ZTF compressed-cutout decoding stays available for BOOM packets.

## Import examples

```python
from al_pipeline.core.loop import ActiveLearningLoop
from al_pipeline.core.scoring import EntropyScorer
from al_pipeline.core.selection import TopKSelector
from al_pipeline.core.pool import Pool
from al_pipeline.core.workflow import ActiveLearningResolutionWorkflow

from al_pipeline.astronomy.boom.source import BoomZTFSource
from al_pipeline.astronomy.boom.evidence import BoomCatalogEvidenceProvider
from al_pipeline.astronomy.labels.applecider_catalog import AppleCiDErCatalogResolver
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer
from al_pipeline.astronomy.experiments.astrominn import build_astrominn_simulated_suite
```

An old facade import may need two new imports: generic oracle types come from
`core.oracle`; catalogue rules come from `astronomy.labels.applecider_catalog`.
This is intentional and keeps the dependency direction visible.

## Validation status on Delta

Completed on 2026-10-08 in the cloned `al-pipeline` environment:

- 316 tests passed in 29.38 seconds.
- A's original and extracted holdout rows and row order matched.
- B's initial labelled rows, hidden labels, and candidate rows/order matched.
  Only its suite-specific candidate `pool_id` differed, as intended.
- B's historical training, validation, and test source hashes matched.
- GPU suite `al_pipeline_smoke_a_v1` completed one entropy cycle with 50
  acquisitions, warm-start training, and one training epoch.
- GPU suite `al_pipeline_smoke_b_v1` completed one bootstrap epoch and one
  entropy cycle with 473 acquisitions and one cycle-training epoch.

These checks establish data-split identity and end-to-end execution of the
extracted A/B pipelines. They do not establish full-budget numerical parity,
old/new fixed-sample logits equality, or recovery after a forced interruption.
The successful Delta sessions used the C++ runtime helper documented in
[Delta installation](delta_installation.md). No external dataset, checkpoint,
or legacy model-code path was changed to achieve these checks.

## Later validation and notebook work

1. When numerical reproduction is needed, compare preprocessing and logits
   on a fixed real sample set with the old code, then use full-budget suites.
2. When recovery testing is needed, interrupt and resume a dedicated check
   suite with unchanged inputs and training policy.
3. Adapt notebooks one at a time using the import maps.

Real-data logits and GPU pilots require Delta's external data and legacy model
files. Passing local synthetic tests alone does not establish full numerical
parity on those external artifacts.
