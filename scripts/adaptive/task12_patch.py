#!/usr/bin/env python3
"""Apply Task12 BALD integration only to a pristine frozen Task4 source copy."""
from pathlib import Path
import shutil


def replace_once(s, a, b, name):
    if s.count(a) != 1:
        raise ValueError(f'Task12 patch anchor {name}: found {s.count(a)}, expected one')
    return s.replace(a, b, 1)


def patch_source(root: Path, overlay: Path):
    root, overlay = Path(root), Path(overlay)
    paths = [
        'src/al_pipeline/core/experiments/config.py',
        'src/al_pipeline/astronomy/experiments/astrominn.py',
        'src/al_pipeline/astronomy/experiments/simulated.py',
        'scripts/run_astrominn_closed_loop.py',
    ]
    for name in paths:
        if not (root/name).is_file(): raise FileNotFoundError(root/name)
    for rel in ('src/al_pipeline/core/scoring/bald.py',
                'src/al_pipeline/astronomy/analysis/task12_mc_dropout.py'):
        (root/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(overlay/rel, root/rel)
    analysis_init = root/'src/al_pipeline/astronomy/analysis/__init__.py'
    analysis_init.parent.mkdir(parents=True, exist_ok=True)
    if not analysis_init.is_file(): analysis_init.write_text('"""Optional read-only analysis modules."""\n')

    p = root/paths[0];s=p.read_text()
    s=replace_once(s,'StrategyName: TypeAlias = Literal[\n',
        'StrategyName: TypeAlias = Literal[\n    "bald",\n','strategy-literal')
    s=replace_once(s,'        if self.strategy not in {\n',
        '        if self.strategy not in {\n            "bald",\n','strategy-validation')
    p.write_text(s)

    p=root/paths[1];s=p.read_text()
    s=replace_once(s,'            "entropy_balanced_explore",\n        }',
        '            "entropy_balanced_explore",\n            "bald",\n        }','suite-strategy')
    s=replace_once(s,
        '    evaluation_manifests = {\n        "validation": dataset.validation_manifest,\n        "test": dataset.test_manifest,\n    }',
        '    evaluation_manifests = {\n        "validation": dataset.validation_manifest,\n    }','evaluation-validation-only')
    p.write_text(s)

    p=root/paths[2];s=p.read_text()
    s=replace_once(s,'from al_pipeline.core.scoring.entropy import EntropyScorer\n',
        'from al_pipeline.core.scoring.entropy import EntropyScorer\n'
        'from al_pipeline.core.scoring.bald import BALDInput, BALDScorer\n'
        'from al_pipeline.astronomy.analysis.task12_mc_dropout import sample_pool\n',
        'imports')
    s=replace_once(s,'        if self.config.strategy == "fisher":\n            return FisherScorer(include_bias=True)\n',
        '        if self.config.strategy == "bald":\n            return BALDScorer()\n'
        '        if self.config.strategy == "fisher":\n            return FisherScorer(include_bias=True)\n',
        'scorer')
    s=replace_once(s,'        if self.config.strategy == "fisher":\n            if inference.embeddings is None:\n',
        '        if self.config.strategy == "bald":\n'
        '            return self._task12_bald_input(inference, cycle)\n'
        '        if self.config.strategy == "fisher":\n'
        '            if inference.embeddings is None:\n',
        'scoring-input')
    anchor='    def _confirmed_object_counts(self, through_cycle: int) -> np.ndarray:\n'
    injected='''    def _task12_bald_input(self, inference: ClassificationInferenceResult, cycle: int) -> BALDInput:
        """Cache all 20 common-model posterior draws, with source provenance.

        This cache is training-independent and contains no candidate labels.
        A score is never accepted with a mismatched checkpoint, IDs or draw plan.
        """
        import hashlib
        path = self.artifact_dir / f"cycle_{cycle:03d}" / "task12_mc_probabilities.npz"
        identity_path = path.with_suffix('.json')
        checkpoint = self._inference_checkpoint(cycle)
        identity = {
            'checkpoint_sha256': _sha256_file(checkpoint),
            'object_ids_sha256': hashlib.sha256(json.dumps(list(inference.object_ids),
                separators=(',', ':')).encode()).hexdigest(),
            'mc_draws': 20, 'seed': self.config.seed, 'cycle': cycle,
            'mask_protocol': 'fixed_per_ViT_dropout_layer_across_all_objects',
            'latest_alert_only': True,
        }
        if path.is_file():
            if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
                raise ValueError('Persisted MC dropout draws have incompatible provenance')
            with np.load(path, allow_pickle=False) as payload:
                ids = tuple(payload['object_ids'].astype(str))
                p = payload['probabilities'].astype(np.float64)
            if ids != tuple(inference.object_ids):
                raise ValueError('MC dropout cache object order differs')
            return BALDInput(p)
        result = sample_pool(
            pool_path=self.pool_path,
            object_ids=tuple(inference.object_ids),
            checkpoint_path=checkpoint,
            artifact_dir=self.predictor.artifact_dir,
            device=self.predictor.device,
            samples=20, batch_size=self.predictor.batch_size,
            seed=self.config.seed, cycle=cycle,
            num_workers=self.predictor.num_workers,
            legacy_code_dir=self.predictor.legacy_code_dir,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f, object_ids=np.asarray(result.object_ids,dtype=str),
                probabilities=result.probabilities)
        temporary.replace(path)
        _json_dump_atomic({**identity,'dropout_layers':list(result.dropout_layers),
            'mean_bald_nats':float(result.scores.mean()),
            'max_bald_nats':float(result.scores.max())}, identity_path)
        # Metadata has supplemental fields; validate only core identity on resume.
        return BALDInput(result.probabilities)

'''
    # Modify identity verification to tolerate diagnostics fields, but not changes to core provenance.
    injected=injected.replace('json.loads(identity_path.read_text()) != identity',
        'any(json.loads(identity_path.read_text()).get(key)!=value for key,value in identity.items())')
    s=replace_once(s,anchor,injected+anchor,'mc-cache-method')
    # Existing lazy score caching is adapted to avoid loading all MC samples on a
    # restarted run where the original BALD scores are already persisted.
    s=replace_once(s,
        '''            scorer = self._scorer(cycle)
            acquisition_data = self._scoring_input(inference, cycle=cycle)
            score_path = path.with_name("candidate_scores.npz")''',
        '''            scorer = self._scorer(cycle)
            score_path = path.with_name("candidate_scores.npz")
            acquisition_data = (
                None if self.config.strategy == "bald" and score_path.exists()
                else self._scoring_input(inference, cycle=cycle)
            )''','avoid-unneeded-MC-cache-load')
    s=replace_once(s,
        '            if self.config.strategy == "entropy_balanced_explore":\n                selection.metadata.update(\n',
        '''            if self.config.strategy == "bald":
                selection.metadata.update({
                    'score_semantics': 'BALD_mutual_information_nats',
                    'mc_dropout_samples': 20,
                    'posterior_approximation': 'coherent_fixed_ViT_dropout_masks',
                    'selection': 'descending_BaldMI_top_k',
                })
            if self.config.strategy == "entropy_balanced_explore":
                selection.metadata.update(
''', 'selection-metadata')
    p.write_text(s)

    p=root/paths[3];s=p.read_text()
    s=replace_once(s,'            "entropy_balanced_explore",\n        ),',
        '            "entropy_balanced_explore",\n            "bald",\n        ),','runner-choices')
    s=replace_once(s,'    args = parse_args()\n',
        '    args = parse_args()\n'
        '    if set(args.strategies) != {"bald"}:\n'
        '        raise ValueError("Task12 runner supports BALD only")\n','exclusive-runner')
    p.write_text(s)
    return paths
