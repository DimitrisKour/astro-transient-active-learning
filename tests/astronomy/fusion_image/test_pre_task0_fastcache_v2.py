"""GPU-free validation of fast pre-Task 0 isolated overlay."""
from __future__ import annotations
import ast
from pathlib import Path

BASE=Path(__file__).resolve().parents[3]
RUNNER=BASE/'scripts/archive/fusion_image/fastcache_v2/fusion_pre_task0_fast.py'
TRAINER=BASE/'src/al_pipeline/astronomy/fusion_image/recipe_training_fast.py'
LAUNCHER=BASE/'scripts/archive/fusion_image/fastcache_v2/submit_fusion_pre_task0_fast.sh'

def test_isolated_fastcache_paths_and_full_history():
    s=RUNNER.read_text()
    assert "new_image_pre_task0_recipe_fastcache_v2" in s
    assert "new_image_full_historical_fastcache_hdd_v1" in s
    assert "FastHistoricalDataset" in TRAINER.read_text()
    assert "ShardedHistoricalNPZDataset" not in TRAINER.read_text()
    assert "'train':29161,'validation':14148" in s
    assert "SOURCE_CACHE" in s
    assert 'test_access' in s and 'acquisition' in s
    assert "fast_cache_manifest.json" in s
    assert "source_cache_manifest_sha256" in s

def test_dataset_no_preload_and_split_separation():
    t=ast.parse(TRAINER.read_text())
    constructors=[]
    for node in ast.walk(t):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='FastHistoricalDataset':
            kwargs={x.arg:x.value for x in node.keywords}
            constructors.append(kwargs)
            assert 'preload' not in kwargs
            assert kwargs['label_column'].id=='label_col'
    assert len(constructors)==2
    assert sorted(node['split'].value for node in constructors)==['train','validation']

def test_independent_submission_requires_first_fit():
    s=LAUNCHER.read_text()
    assert '--submit-case' in s
    assert 'constant/training_summary.json' in s
    assert 'submitted_jobs.tsv' in s
    assert 'new_image_pre_task0_recipe_fastcache_v2' in s

def test_training_records_setup_profile_and_both_checkpoints():
    s=TRAINER.read_text()
    for x in ('model_init_seconds','dataset_init_seconds','manifest_load_seconds','setup_timing.json','best_vit_tower.pth','best_object_f1.pth'):
        assert x in s
