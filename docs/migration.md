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

## Validation order

1. Install in a cloned Delta environment; run the retained tests.
2. Compare A/B preflight parameters, counts, and split hashes using fresh suite IDs.
3. Compare preprocessing and logits on a fixed real sample set with the old code.
4. Run one cycle on a compute node, then verify resume behavior in that new suite.
5. Adapt notebooks one at a time using the import maps.

Real-data logits and GPU pilots require Delta's external data and legacy model
files. Passing local synthetic tests does not establish full numerical parity
on those external artifacts.
