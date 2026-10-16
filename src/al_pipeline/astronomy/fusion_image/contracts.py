"""Strict contracts shared by the newer image/metadata AstroMiNN adapters."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

# Duplicated as an immutable cross-model contract. Delta tests check this
# against the historical ViT constants before any training is allowed.
CLASS_NAMES = ('AGN', 'TDE', 'SN I', 'SN II', 'CV')
LABEL_INDEX = {
    'AGN': 0, 'Nuclear': 0,
    'TDE': 1, 'Tidal Disruption Event': 1,
    'SN I': 2, 'SN Ia': 2, 'SN Ib': 2, 'SN Ic': 2,
    'SN II': 3, 'SN IIn': 3, 'SN IIp': 3, 'SN IIP': 3,
    'CV': 4, 'Cataclysmic': 4,
}


def load_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f'No CSV header: {path}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    return rows


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def label_index(raw: str) -> int:
    try:
        return int(LABEL_INDEX[str(raw).strip()])
    except KeyError as exc:
        raise ValueError(f'Unsupported historical five-class label: {raw!r}') from exc


def class_order_config() -> dict[str, list[str]]:
    """Exact historical training class order, explicitly overriding upstream six classes."""
    return {
        'AGN': ['AGN'],
        'TDE': ['Tidal Disruption Event', 'TDE'],
        'SN I': ['SN Ia', 'SN Ib', 'SN Ic', 'SN I'],
        'SN II': ['SN II', 'SN IIP', 'SN IIn', 'SN IIp'],
        'CV': ['Cataclysmic', 'CV'],
    }


def object_labels(rows: Iterable[Mapping[str, str]], *, id_col: str, label_col: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        oid = str(row[id_col]).strip()
        if not oid:
            raise ValueError('Empty object ID')
        y = label_index(str(row[label_col]))
        if oid in labels and labels[oid] != y:
            raise ValueError(f'Conflicting labels for {oid}')
        labels[oid] = y
    return labels


def load_index(path: str | Path, *, root: str | Path | None = None) -> dict[str, Path]:
    """Map new NPZ object IDs to per-object files; reject duplicate IDs/paths."""
    path = Path(path)
    basedir = Path(root) if root is not None else path.parent
    rows = load_rows(path)
    index: dict[str, Path] = {}
    paths: set[Path] = set()
    for row in rows:
        oid = (row.get('object_id') or row.get('obj_id') or '').strip()
        raw = (row.get('filepath') or '').strip()
        if not oid or not raw:
            raise ValueError(f'New NPZ index requires object_id/obj_id and filepath: {path}')
        p = Path(raw)
        if not p.is_absolute():
            p = basedir / p
        if oid in index or p in paths:
            raise ValueError(f'Duplicate new NPZ object/path: {oid} {p}')
        index[oid] = p
        paths.add(p)
    return index


def compare_splits(*, historical: Mapping[str, Path], new_index: Mapping[str, Path],
                   new_splits: Mapping[str, Path] | None = None,
                   new_fine_labels: Mapping[str, str] | None = None,
                   id_col: str = 'obj_id', label_col: str = 'target') -> dict[str, Any]:
    """Read-only audit: coverage, label agreement, and split membership, no silent filtering."""
    old_by_split: dict[str, dict[str, int]] = {}
    old_all: set[str] = set()
    results: dict[str, Any] = {'historical_splits': {}, 'new_index_objects': len(new_index)}
    for split, path in historical.items():
        records = load_rows(path)
        labels = object_labels(records, id_col=id_col, label_col=label_col)
        ids = set(labels)
        overlapping = ids & old_all
        if overlapping:
            raise ValueError(f'Historical splits overlap: {split} ({len(overlapping)})')
        old_all.update(ids)
        old_by_split[split] = labels
        shared = ids & set(new_index)
        missing = ids - set(new_index)
        results['historical_splits'][split] = {
            'objects': len(ids), 'sample_rows': len(records),
            'matched_new_npz': len(shared), 'missing_new_npz': len(missing),
            'missing_examples': sorted(missing)[:20],
            'class_counts': {CLASS_NAMES[k]: v for k, v in sorted(Counter(labels.values()).items())},
            'manifest_sha256': sha256(path),
        }
    results['full_historical_npz_coverage'] = all(
        d['missing_new_npz'] == 0 for d in results['historical_splits'].values()
    )
    if new_fine_labels is not None:
        mismatches = []
        missing_new_labels = []
        checked_labels = 0
        for split, labels in old_by_split.items():
            for oid, expected in labels.items():
                if oid not in new_index:
                    continue
                if oid not in new_fine_labels or not new_fine_labels[oid]:
                    missing_new_labels.append(oid)
                    continue
                try:
                    observed = label_index(new_fine_labels[oid])
                except ValueError:
                    observed = -1
                checked_labels += 1
                if observed != expected:
                    mismatches.append({'object_id':oid,'historical':CLASS_NAMES[expected],
                                       'new_fine_label':new_fine_labels[oid]})
        results['fine_label_audit'] = {
            'checked':checked_labels, 'missing':len(missing_new_labels),
            'mismatches':len(mismatches), 'mismatch_examples':mismatches[:20],
            'missing_examples':missing_new_labels[:20],
        }
        results['all_historical_labels_verified'] = (
            results['full_historical_npz_coverage'] and
            not mismatches and not missing_new_labels
        )
    else:
        results['all_historical_labels_verified'] = False
    if new_splits:
        per_split: dict[str, set[str]] = {}
        for split, path in new_splits.items():
            r = load_rows(path)
            ids = {x.get('object_id') or x.get('obj_id') for x in r}
            if None in ids:
                raise ValueError(f'Missing IDs in new split: {path}')
            per_split[split] = ids
        results['new_split_assignment'] = {}
        for oldsplit, ids in old_by_split.items():
            results['new_split_assignment'][oldsplit] = {
                newsplit: len(set(ids) & new_ids)
                for newsplit, new_ids in per_split.items()
            }
        results['identical_split_membership'] = all(
            set(old_by_split[s]) == per_split[s]
            for s in old_by_split if s in per_split
        ) and set(old_by_split) == set(per_split)
    return results


def write_json_atomic(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(p)
