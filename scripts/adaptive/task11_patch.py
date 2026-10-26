#!/usr/bin/env python3
"""Strict, frozen-source-only Task11 AstroMiNN acquisition integration.

Runs *only* against a copied source tree. An anchor mismatch fails closed;
Task4, other policies and live checkout remain untouched.
"""
from pathlib import Path
import shutil

POLICIES = ('badge', 'core_set', 'fisher_batch')


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f'Task11 patch anchor mismatch [{label}]: expected 1, found {source.count(old)}')
    return source.replace(old,new,1)


def patch_source(root: Path, overlay: Path) -> list[str]:
    root,overlay=Path(root),Path(overlay)
    targets = [
        'src/al_pipeline/core/experiments/config.py',
        'src/al_pipeline/astronomy/experiments/astrominn.py',
        'src/al_pipeline/astronomy/experiments/simulated.py',
        'scripts/run_astrominn_closed_loop.py',
    ]
    for path in targets:
        if not (root/path).is_file():
            raise FileNotFoundError(root/path)
    for dest in [
        'src/al_pipeline/core/advanced_acquisition/fisher_batch.py',
        'src/al_pipeline/astronomy/analysis/task11_labeled_features.py',
    ]:
        (root/dest).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(overlay/dest,root/dest)

    p=root/targets[0];s=p.read_text()
    s=replace_once(s,'StrategyName: TypeAlias = Literal[\n','StrategyName: TypeAlias = Literal[\n    "badge",\n    "core_set",\n    "fisher_batch",\n','config-literal')
    s=replace_once(s,'        if self.strategy not in {\n','        if self.strategy not in {\n            "badge", "core_set", "fisher_batch",\n','config-validator')
    p.write_text(s)

    p=root/targets[1];s=p.read_text()
    s=replace_once(s,'            "entropy_balanced_explore",\n        }','            "entropy_balanced_explore",\n            "badge", "core_set", "fisher_batch",\n        }','suite-strategy')
    p.write_text(s)

    # Task11 never computes repeated held-out test metrics during policy development.
    p=root/targets[1];s=p.read_text()
    s=replace_once(s,
'''    evaluation_manifests = {
        "validation": dataset.validation_manifest,
        "test": dataset.test_manifest,
    }''',
'''    evaluation_manifests = {
        "validation": dataset.validation_manifest,
    }''','validation-only-evaluation')
    p.write_text(s)

    p=root/targets[2];s=p.read_text()
    s=replace_once(s,'from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector\n',
'''from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector
from al_pipeline.core.advanced_acquisition.badge import BadgeInput, BadgeGradientNormScorer, BadgeSelector
from al_pipeline.core.advanced_acquisition.core_set import CoreSetInput, CoreSetScorer, CoreSetSelector
from al_pipeline.core.advanced_acquisition.fisher_batch import FisherBatchInput, FisherBatchScorer, FisherBatchSelector
from al_pipeline.astronomy.analysis.task11_labeled_features import predict_confirmed_objects
''','simulation-imports')
    s=replace_once(s,'            if self.config.strategy == "entropy_diversity":\n',
'''            if self.config.strategy == "badge":
                selector = BadgeSelector(acquisition_data, seed=int(np.random.SeedSequence(
                    [self.config.seed,cycle,11]).generate_state(1)[0]))
            elif self.config.strategy == "core_set":
                selector = CoreSetSelector(acquisition_data)
            elif self.config.strategy == "fisher_batch":
                selector = FisherBatchSelector(acquisition_data)
            elif self.config.strategy == "entropy_diversity":
''','simulation-select')
    s=replace_once(s,'        if self.config.strategy == "fisher":\n            return FisherScorer(include_bias=True)\n',
'''        if self.config.strategy == "badge":
            return BadgeGradientNormScorer()
        if self.config.strategy == "core_set":
            return CoreSetScorer()
        if self.config.strategy == "fisher_batch":
            return FisherBatchScorer()
        if self.config.strategy == "fisher":
            return FisherScorer(include_bias=True)
''','simulation-scorer')
    s=replace_once(s,'        if self.config.strategy == "fisher":\n            if inference.embeddings is None:\n',
'''        if self.config.strategy == "badge":
            if inference.embeddings is None:
                raise ValueError('BADGE requires classifier-input embeddings')
            return BadgeInput(
                predictions=inference.predictions,embeddings=inference.embeddings,
                candidate_ids=tuple(inference.object_ids),include_bias=True)
        if self.config.strategy in {"core_set", "fisher_batch"}:
            if inference.embeddings is None:
                raise ValueError('Feature-based acquisition requires embeddings')
            anchor = self._task11_confirmed_inference(cycle)
            if anchor.embeddings is None:
                raise ValueError('Confirmed objects have no embeddings')
            if self.config.strategy == "core_set":
                def unit(x):
                    x=np.asarray(x,dtype=np.float64)
                    norm=np.linalg.norm(x,axis=1,keepdims=True)
                    return x/np.maximum(norm,1e-12)
                return CoreSetInput(
                    candidate_embeddings=unit(inference.embeddings),
                    labeled_embeddings=unit(anchor.embeddings),
                    candidate_ids=tuple(inference.object_ids),
                    labeled_ids=tuple(anchor.object_ids))
            return FisherBatchInput(
                candidate_ids=tuple(inference.object_ids),
                probabilities=inference.predictions.as_probabilities(),
                embeddings=inference.embeddings,
                labeled_probabilities=anchor.predictions.as_probabilities(),
                labeled_embeddings=anchor.embeddings,
                rank=32,ridge=1.0,projection_seed=271828)

        if self.config.strategy == "fisher":
            if inference.embeddings is None:
''','simulation-scoring-input')
    s=replace_once(s,'    def _confirmed_object_counts(self, through_cycle: int) -> np.ndarray:\n',
'''    def _task11_confirmed_inference(self, cycle: int) -> ClassificationInferenceResult:
        """Read only confirmed L_(cycle-1), never current hidden candidate labels."""
        confirmed = self.labeled_sets.training_manifest(cycle-1)
        checkpoint = self._inference_checkpoint(cycle)
        cache = self.artifact_dir / f"cycle_{cycle:03d}" / "task11_confirmed_inference.npz"
        identity = cache.with_suffix('.json')
        fingerprint = {
            'manifest_sha256':_sha256_file(confirmed),
            'checkpoint_sha256':_sha256_file(checkpoint),
        }
        if cache.is_file():
            if not identity.is_file() or json.loads(identity.read_text()) != fingerprint:
                raise ValueError('Confirmed inference cache provenance changed')
            return _load_inference(cache)
        result=predict_confirmed_objects(
            manifest_path=confirmed,dataset=self.dataset,
            checkpoint_path=checkpoint,artifact_dir=self.predictor.artifact_dir,
            device=self.predictor.device,batch_size=self.predictor.batch_size,
            num_workers=self.predictor.num_workers,
            legacy_code_dir=self.predictor.legacy_code_dir)
        _json_dump_atomic(fingerprint,identity)
        _save_inference(result,cache)
        return result

    def _confirmed_object_counts(self, through_cycle: int) -> np.ndarray:
''','simulation-anchor-infer')
    p.write_text(s)

    p=root/targets[3];s=p.read_text()
    s=replace_once(s,'            "entropy_balanced_explore",\n        ),','            "entropy_balanced_explore",\n            "badge", "core_set", "fisher_batch",\n        ),','runner-choices')
    s=replace_once(s,'    args = parse_args()\n','    args = parse_args()\n    if set(args.strategies) - {"badge", "core_set", "fisher_batch"}:\n        raise ValueError("Task11 runner permits only the three frozen Task11 acquisition policies")\n','runner-exclusive')
    p.write_text(s)

    # Accelerate the labeled-anchor k-center initialization and reuse scores.
    p=root/'src/al_pipeline/core/advanced_acquisition/core_set.py';s=p.read_text()
    s=replace_once(s,'from dataclasses import dataclass\n','from dataclasses import dataclass, field\n','coreset-field-import')
    s=replace_once(s,'    labeled_ids: tuple[str, ...]\n','    labeled_ids: tuple[str, ...]\n    _task11_anchor_distances: np.ndarray | None = field(default=None,init=False,repr=False)\n','coreset-cache-attribute')
    s=replace_once(s,
'''        distances = np.full(self.candidate_embeddings.shape[0], np.inf)
        for anchor in self.labeled_embeddings:
            distances = np.minimum(distances, squared_distances(self.candidate_embeddings, anchor))
        return distances''',
'''        cached = getattr(self, "_task11_anchor_distances", None)
        if cached is not None:
            return cached.copy()
        from scipy.spatial.distance import cdist
        result = np.empty(len(self.candidate_embeddings), dtype=np.float64)
        for first in range(0,len(result),256):
            stop = min(first+256,len(result))
            dist = cdist(self.candidate_embeddings[first:stop],
                         self.labeled_embeddings,metric='sqeuclidean')
            result[first:stop] = dist.min(axis=1)
        object.__setattr__(self,'_task11_anchor_distances',result)
        return result.copy()''','coreset-cdist')
    p.write_text(s)
    return targets + ['src/al_pipeline/core/advanced_acquisition/core_set.py']
