"""Provenance checks and exact grouped NPZ stage extraction for BTSbot cache v2.

The original per-alert production loader remains the numerical reference.
This helper does not cache labels, or use the newer model's training taxonomy.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .contracts import sha256
from .data import prepare_canonical_image


def _upstream_files() -> tuple[Path, Path]:
    import fusion_model_basics
    root = Path(fusion_model_basics.__file__).resolve().parent
    paths = (root / 'dataset.py', root / 'preprocessing.py')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def preprocessing_identity(config: dict) -> dict:
    """Only config fields actually read by the deterministic image transform."""
    ds_path, prep_path = _upstream_files()
    return {
        'image_encoder_type': str(config.get('image_encoder_type', 'patch')).lower(),
        'image_norm': str(config.get('image_norm', 'median')),
        'patch_size': list(config.get('patch_size', [49, 7])),
        'cutout_size': int(config.get('cutout_size', 63)),
        'upstream_dataset_sha256': sha256(ds_path),
        'upstream_preprocessing_sha256': sha256(prep_path),
    }


def read_object_stages(path: Path, alerts: set[int], config: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read one NPZ, build all 55 features ONCE, and select verified alert indices."""
    from fusion_model_basics.dataset import AlertStreamFusionDataset, _as_str_list
    with np.load(path, allow_pickle=True) as z:
        required = {'images', 'meta_data', 'meta_columns', 'jd', 'fid'}
        if not required.issubset(z.files):
            raise ValueError(f'Missing NPZ fields {sorted(required-set(z.files))}: {path}')
        images = np.asarray(z['images'], dtype=np.float32)
        meta = np.asarray(z['meta_data'], dtype=np.float32)
        cols = _as_str_list(z['meta_columns'])
        jd = np.asarray(z['jd'], dtype=np.float64)
        fid = np.asarray(z['fid'], dtype=np.int16)
        prov = np.asarray(z['provenance'], dtype=object) if 'provenance' in z.files else None
    n = len(images)
    if n < 1 or images.shape[1:] != (3,63,63) or len(meta) != n or len(jd) != n or len(fid) != n:
        raise ValueError(f'Malformed NPZ stage arrays: {path}')
    if prov is not None and len(prov) != n:
        raise ValueError(f'Malformed NPZ provenance: {path}')
    alert_locations: dict[int, list[int]] = defaultdict(list)
    for k in range(n):
        number = k
        if prov is not None:
            entry = prov[k]
            if isinstance(entry, Mapping) and entry.get('alert_idx') is not None:
                idx = int(entry['alert_idx'])
                if idx >= 0:
                    number = idx
        alert_locations[number].append(k)
    with torch.no_grad():
        metadata = AlertStreamFusionDataset._build_metadata_tensor(None, meta, cols, jd, fid, prov)
        if tuple(metadata.shape) != (n,55):
            raise ValueError(f'Expected (N,55) upstream metadata in {path}; got {tuple(metadata.shape)}')
        out = {}
        for alert in sorted(alerts):
            locations = alert_locations.get(int(alert), [])
            if len(locations) != 1:
                raise ValueError(f'Alert {alert} has {len(locations)} matches in {path}')
            k = locations[0]
            image = prepare_canonical_image(images[k], runtime_config=config, augment=False)
            feature = metadata[k]
            if not torch.isfinite(image).all() or not torch.isfinite(feature).all():
                raise ValueError(f'Non-finite input for alert {alert}: {path}')
            out[alert] = (feature.detach().cpu().numpy().astype(np.float32,copy=False),
                          image.detach().cpu().numpy().astype(np.float32,copy=False))
    return out
