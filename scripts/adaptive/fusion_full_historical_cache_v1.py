#!/usr/bin/env python3
"""Preprocess the COMPLETE historical training population + fixed validation.

Strictly CPU-only. No training, no test access, no oracle calls, no hidden-label
exposure in cache shards. Uses the already-tested sharded cache v2 engine.
This is different from Experiment B's small initially labeled AL L0 subset.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image import cache_helpers, cache_v2, data
from al_pipeline.astronomy.fusion_image.cache_v2 import prepare, build_shard, finalize
from al_pipeline.astronomy.fusion_image.contracts import (
    class_order_config, label_index, load_rows, object_labels, sha256, write_json_atomic,
)

CACHE_SOURCE = 'historical_full_train_plus_fixed_validation'


def _verify_source(source_plan: Path):
    """Reject mismatched split, audit, NPZ catalog or upstream checkout."""
    prior = json.loads(source_plan.read_text(encoding='utf-8'))
    for required in ('upstream_repo', 'model_sha', 'audit', 'new_index',
                     'registry', 'file_hashes', 'l0_manifest',
                     'historical_train_objects', 'l0_rows', 'hf_snapshot'):
        if required not in prior:
            raise ValueError(f'Missing field {required} in original L0 matrix plan')
    upstream_sha = subprocess.check_output(
        ['git', '-C', prior['upstream_repo'], 'rev-parse', 'HEAD'],
        text=True,
    ).strip()
    if upstream_sha != prior['model_sha']:
        raise ValueError('Upstream model checkout changed since audited plan')
    for key in ('audit', 'new_index', 'registry', 'l0_manifest'):
        if sha256(prior[key]) != prior['file_hashes'][key]:
            raise ValueError(f'Changed original L0 plan input: {key}')
    ds = DatasetRegistry.from_toml(prior['registry']).resolve_profile('astrominn_ztf').dataset
    if ds.train_manifest is None or ds.validation_manifest is None:
        raise ValueError('Historical fixed train and validation manifests required')
    if not ds.object_id_column or not ds.label_column:
        raise ValueError('Missing object/label column in historical registry')
    manifests = {'train': Path(ds.train_manifest), 'validation': Path(ds.validation_manifest)}
    for split, p in manifests.items():
        if sha256(p) != prior['file_hashes'][f'{split}_manifest']:
            raise ValueError(f'Historical FULL {split} manifest differs from prior audit')
    audit = json.loads(Path(prior['audit']).read_text(encoding='utf-8'))
    for key in ('full_historical_npz_coverage', 'all_historical_labels_verified',
                'all_historical_stages_verified', 'stage_probe_passed'):
        if audit.get(key) is not True:
            raise ValueError(f'Historical complete-dataset audit not passed: {key}')
    if audit['new_index_sha256'] != prior['file_hashes']['new_index']:
        raise ValueError('Audited NPZ index differs from the L0 plan')
    rows = {split: load_rows(path) for split, path in manifests.items()}
    objs = {split: object_labels(rows[split], id_col=ds.object_id_column,
                   label_col=ds.label_column) for split in manifests}
    if set(objs['train']) & set(objs['validation']):
        raise ValueError('Train/validation object leakage')
    if len(objs['train']) != prior['historical_train_objects']:
        raise ValueError('Historical full training object count mismatch')
    if len(rows['train']) <= prior['l0_rows']:
        raise ValueError('Full-history train must be larger than AL initial L0')
    for split in manifests:
        if audit['historical_splits'][split]['sample_rows'] != len(rows[split]):
            raise ValueError(f'Audit row count disagrees with {split} data')
        stage = audit['stage_checks'][split]
        if stage['failed_rows'] or stage['checked_rows'] != len(rows[split]) or \
           stage['compatible_rows'] != len(rows[split]):
            raise ValueError(f'Incomplete audited alert stages for {split}')
    # Verify small original Experiment B L0 is a subset; do not change it.
    old_rows = load_rows(prior['l0_manifest'])
    def triplets(contents):
        return Counter((r[ds.object_id_column], int(r['alert']),
                        label_index(r[ds.label_column])) for r in contents)
    if triplets(old_rows) - triplets(rows['train']):
        raise ValueError('Old AL initial L0 is not contained in historical full train')
    return prior, ds, manifests, rows, objs


def _runtime(prior):
    # Same upstream build_astrominn_config call as successful cache v3.
    from fusion_model_basics.config import build_astrominn_config
    return build_astrominn_config(
        'btsbot_images', coord_checkpoint=None,
        hf_image_model_dir=prior['hf_snapshot'],
        classes=class_order_config(), device='cpu', dir='.',
    )


def _identity(source_plan, cache_root, prior, manifests, rows, objs):
    return {
        'scope': CACHE_SOURCE,
        'source_matrix_plan': str(source_plan.resolve()),
        'source_matrix_plan_sha256': sha256(source_plan),
        'cache_plan_sha256': sha256(cache_root/'plan.json'),
        'full_train_manifest': str(manifests['train'].resolve()),
        'full_train_manifest_sha256': sha256(manifests['train']),
        'validation_manifest': str(manifests['validation'].resolve()),
        'validation_manifest_sha256': sha256(manifests['validation']),
        'previous_small_al_l0_manifest': str(Path(prior['l0_manifest']).resolve()),
        'previous_small_al_l0_manifest_sha256': sha256(prior['l0_manifest']),
        'training_objects': len(objs['train']),
        'training_rows': len(rows['train']),
        'validation_objects': len(objs['validation']),
        'validation_rows': len(rows['validation']),
        'model_source_sha': prior['model_sha'],
        'cache_implementation_sha256': {
            'cache_v2': sha256(cache_v2.__file__),
            'cache_helpers': sha256(cache_helpers.__file__),
            'data': sha256(data.__file__),
            'full_cache_runner': sha256(__file__),
        },
        'labels_saved_in_cache': False,
        'test_split_accessed': False,
    }


def action(command: str, *, source_plan: Path, cache_root: Path,
           n_shards: int = 24, shard_id: int | None = None):
    source_plan = source_plan.resolve()
    cache_root = cache_root.resolve()
    prior, ds, manifests, rows, objs = _verify_source(source_plan)
    cfg = _runtime(prior)
    if command == 'prepare':
        plan = prepare(
            cache_root=cache_root, manifests=manifests,
            index_csv=Path(prior['new_index']), runtime_config=cfg,
            id_col=ds.object_id_column, label_col=ds.label_column,
            n_shards=n_shards, upstream_sha=prior['model_sha'],
        )
        identity = _identity(source_plan, cache_root, prior, manifests, rows, objs)
        record_path = cache_root/'full_historical_identity.json'
        if record_path.exists():
            if json.loads(record_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Historical cache identity changed since prepare')
        else:
            write_json_atomic(record_path, identity)
        return {'command': command, 'cache_root': str(cache_root),
                'training_objects': len(objs['train']),
                'training_rows': len(rows['train']),
                'validation_rows': len(rows['validation']),
                'previous_small_l0_rows': prior['l0_rows'],
                'unique_alert_stages': plan['unique_alert_stages'],
                'shards': n_shards, 'gpu_jobs_submitted': 0}
    record_path = cache_root/'full_historical_identity.json'
    if not record_path.exists():
        raise FileNotFoundError(f'Run prepare first: {record_path}')
    saved = json.loads(record_path.read_text(encoding='utf-8'))
    if saved != _identity(source_plan, cache_root, prior, manifests, rows, objs):
        raise ValueError('Historical cache identity no longer matches verified source')
    if command == 'build-shard':
        if shard_id is None:
            raise ValueError('build-shard requires shard_id')
        return build_shard(root=cache_root, shard_id=shard_id, runtime_config=cfg)
    if command == 'finalize':
        return finalize(cache_root, source=CACHE_SOURCE)
    if command == 'status':
        done = sum((cache_root/f'shard_{i:03d}'/'done.json').is_file()
                   for i in range(n_shards))
        manifest = cache_root/'cache_manifest.json'
        return {'cache_root': str(cache_root), 'completed_shards': done,
                'expected_shards': n_shards,
                'finalized': manifest.is_file(),
                'cache_source': json.loads(manifest.read_text()).get('source')
                   if manifest.is_file() else None,
                'gpu_jobs_submitted': 0}
    raise ValueError(f'Unsupported operation {command}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'build-shard','finalize','status'))
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=24)
    parser.add_argument('--shard-id', type=int)
    args = parser.parse_args()
    result = action(args.command, source_plan=args.source_plan,
                    cache_root=args.cache_root, n_shards=args.shards,
                    shard_id=args.shard_id)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
