#!/usr/bin/env python3
"""Read-only overlap audit and smoke-manifest builder for new NPZs.

Run from al-pipeline repo root. This does not change original replay data.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.contracts import (
    CLASS_NAMES, compare_splits, label_index, load_index, load_rows, sha256, write_json_atomic,
)
from al_pipeline.astronomy.fusion_image.data import audit_npz_alert_stages, image_metadata_from_npz


def historical_dataset(registry: Path, profile: str):
    return DatasetRegistry.from_toml(registry).resolve_profile(profile).dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', required=True, type=Path)
    ap.add_argument('--new-index', required=True, type=Path, help='new built_all.csv (object_id, filepath)')
    ap.add_argument('--new-root', type=Path, default=None)
    ap.add_argument('--new-split-dir', type=Path, help='Optional new scheme-A split dir; compares split membership only')
    ap.add_argument('--profile', default='astrominn_ztf')
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--check-stages', type=int, default=20,
                    help='Check first N matched historical rows per split; use -1 for all')
    ap.add_argument('--variant', choices=('patch_images', 'btsbot_images'), default='patch_images')
    ap.add_argument('--hf-snapshot', type=Path)
    ap.add_argument('--model-sha', help='Exact git rev-parse HEAD of cloned upstream')
    ap.add_argument('--strict-stages', action='store_true',
                    help='Exit nonzero after writing audit.json unless every historical stage verifies')
    ap.add_argument('--make-smoke', action='store_true', help='Write five-class train+val smoke manifests')
    ap.add_argument('--smoke-per-class', type=int, default=2)
    args = ap.parse_args()

    dataset = historical_dataset(args.registry, args.profile)
    manifests = {'train': dataset.train_manifest, 'validation': dataset.validation_manifest,
                 'test': dataset.test_manifest}
    if not all(p is not None and p.exists() for p in manifests.values()):
        raise FileNotFoundError('Historical train/validation/test manifests required')
    old_new_split = {'train': 'train', 'validation': 'val', 'test': 'test'}
    nsplit = None
    if args.new_split_dir:
        nsplit = {key: args.new_split_dir/f'manifest_{val}.csv'
                  for key, val in old_new_split.items()}
    index = load_index(args.new_index, root=args.new_root)
    index_rows = load_rows(args.new_index)
    fine_labels = {r.get('object_id') or r.get('obj_id'): r.get('label_str', '')
                   for r in index_rows}
    summary = compare_splits(historical=manifests, new_index=index, new_splits=nsplit, new_fine_labels=fine_labels,
                             id_col=dataset.object_id_column,
                             label_col=dataset.label_column)
    summary.update({
        'new_index': str(args.new_index.resolve()),
        'new_index_sha256': sha256(args.new_index),
        'new_root': str(args.new_root.resolve()) if args.new_root is not None else None,
        'new_model_repository_sha': args.model_sha,
        'class_names': list(CLASS_NAMES),
        'note': 'Scheme A labels differ; full training compatibility requires five-class labels and stage-level validation.',
    })
    # Keep the historical row order and first-N sampling rule unchanged.
    # Across splits, each object is opened once and its requested alert stages
    # are checked together. We still account for every original sample row.
    checked = {}
    failures = {}
    rows_by_object: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for split, file in manifests.items():
        rows = [r for r in load_rows(file) if r[dataset.object_id_column] in index]
        if args.check_stages >= 0:
            rows = rows[:args.check_stages]
        checked[split] = {'checked_rows': len(rows), 'compatible_rows': 0, 'failed_rows': 0}
        failures[split] = []
        for row in rows:
            rows_by_object[row[dataset.object_id_column]].append((split, row))

    runtime = {'image_encoder_type': 'patch' if args.variant=='patch_images' else 'btsbot_convnext',
               'patch_size': [49, 7], 'cutout_size': 63,
               'image_norm': 'median' if args.variant=='patch_images' else 'none'}

    io_objects = 0
    unique_stage_requests = 0
    for object_number, (oid, records) in enumerate(rows_by_object.items(), 1):
        wanted: set[int] = set()
        parsed: list[tuple[str, dict[str, str], int | None, str | None]] = []
        for split, row in records:
            try:
                alert = int(row['alert'])
                wanted.add(alert)
                parsed.append((split, row, alert, None))
            except Exception as exc:
                parsed.append((split, row, None, str(exc)))
        unique_stage_requests += len(wanted)
        if wanted:
            io_objects += 1
            outcomes = audit_npz_alert_stages(index[oid], wanted, runtime_config=runtime)
        else:
            outcomes = {}
        for split, row, alert, parse_error in parsed:
            error = parse_error if parse_error is not None else outcomes[alert]
            if error is None:
                checked[split]['compatible_rows'] += 1
            else:
                checked[split]['failed_rows'] += 1
                if len(failures[split]) < 30:
                    failures[split].append({
                        'object_id': oid, 'alert': row.get('alert', ''), 'error': error,
                    })

        if object_number % 250 == 0 or object_number == len(rows_by_object):
            failed_so_far = sum(x['failed_rows'] for x in checked.values())
            print(
                f'[stage audit] objects {object_number}/{len(rows_by_object)}; '
                f'failures {failed_so_far}', flush=True,
            )

    summary['stage_checks'] = checked
    summary['stage_errors'] = failures
    summary['stage_scan'] = {
        'method': 'group_by_object_one_npz_read',
        'unique_historical_objects_checked': len(rows_by_object),
        'npz_files_opened': io_objects,
        'distinct_alert_stages_requested': unique_stage_requests,
    }
    summary['stage_probe_passed'] = all(x['failed_rows'] == 0 for x in checked.values())
    expected_rows = sum(item['sample_rows'] for item in summary['historical_splits'].values())
    actually_checked = sum(item['checked_rows'] for item in checked.values())
    summary['all_historical_stages_verified'] = (
        args.check_stages == -1
        and summary['stage_probe_passed']
        and summary['full_historical_npz_coverage']
        and summary['all_historical_labels_verified']
        and actually_checked == expected_rows
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir/'audit.json', summary)
    if args.make_smoke:
        if not args.model_sha:
            raise ValueError('--make-smoke requires --model-sha for source provenance')
        for split in ('train', 'validation'):
            rows = load_rows(manifests[split])
            count_by_class: dict[int, int] = defaultdict(int)
            seen = set()
            chosen = []
            runtime = {'image_encoder_type': 'patch' if args.variant=='patch_images' else 'btsbot_convnext',
                       'patch_size': [49, 7], 'cutout_size': 63,
                       'image_norm': 'median' if args.variant=='patch_images' else 'none'}
            for row in rows:
                oid = row[dataset.object_id_column]
                y = label_index(row[dataset.label_column])
                if oid in seen or oid not in index or count_by_class[y] >= args.smoke_per_class:
                    continue
                try:
                    image_metadata_from_npz(index[oid], int(row['alert']), runtime_config=runtime)
                except Exception:
                    continue
                count_by_class[y] += 1
                seen.add(oid)
                chosen.append(row)
                if len(chosen) == 5 * args.smoke_per_class:
                    break
            if len(chosen) != 5 * args.smoke_per_class:
                raise ValueError(f'Insufficient stage-matched objects for {split} smoke: {dict(count_by_class)}')
            target = args.out_dir/f'smoke_{split}.csv'
            with target.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(chosen[0]))
                w.writeheader(); w.writerows(chosen)
            print(f'Wrote {target}: {len(chosen)} rows')
    print(json.dumps({k: v for k, v in summary.items() if k != 'stage_errors'}, indent=2))
    print(f'Full audit: {args.out_dir/"audit.json"}')
    if args.strict_stages and not summary['all_historical_stages_verified']:
        raise SystemExit('FAILED: full historical stage verification did not pass; see audit.json')


if __name__ == '__main__':
    main()
