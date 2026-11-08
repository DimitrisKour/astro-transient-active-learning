import ast
import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('task12_admin',ROOT/'scripts/research/historical_replay/task12/task12.py')
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod;spec.loader.exec_module(mod)


def test_plan_scientific_freeze_and_computational_budget():
    p=mod.plan()
    assert p['fits']==27 and p['cycles']==9 and p['posterior_samples']==20
    assert p['source_baseline']=='original_verified_Task4_frozen_source'
    assert p['batchbald'].startswith('deferred')
    assert p['test_used_for_selection'] is False


def test_bald_runner_scoped_and_after_task11_required():
    s=(ROOT/'scripts/research/historical_replay/task12/task12_patch.py').read_text()
    assert 'Task12 runner supports BALD only' in s
    assert 'validation' in s
    assert '"test": dataset.test_manifest' in s  # expected original source anchor, which is removed
    a=(ROOT/'scripts/research/historical_replay/task12/task12.py').read_text()
    assert 'Task11 not yet fully completed or submitted' in a
    assert "'afterok:'" in a
    assert "'--hold'" in a


def test_patch_fails_closed_when_upstream_anchor_missing():
    p=ROOT/'scripts/research/historical_replay/task12/task12_patch.py'
    spec=importlib.util.spec_from_file_location('task12_patch',p)
    s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
    with pytest.raises(ValueError,match='patch anchor'):
        s.replace_once('old old','NOT THERE','new','test')


def test_package_uses_normal_training_settings():
    a=(ROOT/'scripts/research/historical_replay/task12/task12.py').read_text()
    for expected in ('--max-epochs','--patience','--batch-size',"'warm_start'",'--complete-pool'):
        assert expected in a
    for f in (ROOT/'scripts/research/historical_replay/task12').glob('task12*.py'):
        ast.parse(f.read_text())


def test_strict_patch_updates_exact_scientific_boundaries(tmp_path):
    import shutil
    overlay=ROOT
    src=tmp_path
    def put(rel,text):
        target=src/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    put('src/al_pipeline/core/experiments/config.py',
        'StrategyName: TypeAlias = Literal[\n    "entropy",\n]\n'
        '        if self.strategy not in {\n            "entropy",\n        }: pass\n')
    put('src/al_pipeline/astronomy/experiments/astrominn.py',
        '            "entropy_balanced_explore",\n        }\n'
        '    evaluation_manifests = {\n        "validation": dataset.validation_manifest,\n'
        '        "test": dataset.test_manifest,\n    }\n')
    put('src/al_pipeline/astronomy/experiments/simulated.py',
        'from al_pipeline.core.scoring.entropy import EntropyScorer\n'
        '            scorer = self._scorer(cycle)\n'
        '            acquisition_data = self._scoring_input(inference, cycle=cycle)\n'
        '            score_path = path.with_name("candidate_scores.npz")\n'
        '            if self.config.strategy == "entropy_balanced_explore":\n'
        '                selection.metadata.update(\n'
        '        if self.config.strategy == "fisher":\n'
        '            return FisherScorer(include_bias=True)\n'
        '        if self.config.strategy == "fisher":\n'
        '            if inference.embeddings is None:\n'
        '    def _confirmed_object_counts(self, through_cycle: int) -> np.ndarray:\n')
    put('scripts/run_astrominn_closed_loop.py',
        '            "entropy_balanced_explore",\n        ),\n'
        '    args = parse_args()\n')
    q=ROOT/'scripts/research/historical_replay/task12/task12_patch.py'
    spec=importlib.util.spec_from_file_location('task12_patch_smoke',q)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.patch_source(src,overlay)
    cfg=(src/'src/al_pipeline/core/experiments/config.py').read_text()
    runner=(src/'scripts/run_astrominn_closed_loop.py').read_text()
    sim=(src/'src/al_pipeline/astronomy/experiments/simulated.py').read_text()
    suite=(src/'src/al_pipeline/astronomy/experiments/astrominn.py').read_text()
    assert '"bald"' in cfg
    assert 'Task12 runner supports BALD only' in runner
    assert '"test": dataset.test_manifest' not in suite
    assert '_task12_bald_input' in sim and 'score_semantics' in sim
    assert 'task12_mc_dropout.py' in [p.name for p in (src/'src/al_pipeline/astronomy/analysis').glob('*.py')]
    for p in [src/'src/al_pipeline/core/scoring/bald.py',
              src/'src/al_pipeline/astronomy/analysis/task12_mc_dropout.py']:
        ast.parse(p.read_text())
