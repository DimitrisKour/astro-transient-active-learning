"""Load new per-object NPZ data via the upstream's exact metadata/image code.

Source manifest labels and object identities come exclusively from the historical
training/evaluation CSVs, never from the upstream scheme-A label column.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import label_index, load_rows


def prepare_canonical_image(
    image: np.ndarray | torch.Tensor, *, runtime_config: dict,
    augment: bool = False,
) -> torch.Tensor:
    """Use the upstream image transform and assert its model-input shape.

    The stored ZTF cutout is always (3, 63, 63), but `patch_images`
    expects the upstream central 49x49 crop. Passing an unprocessed 63x63
    cutout produces 81 patch tokens against 49 positional embeddings.
    `btsbot_images` instead consumes the full 63x63 cutout.
    """
    from fusion_model_basics.preprocessing import _prepare_astrominn_image

    raw = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
    if raw.shape != (3, 63, 63):
        raise ValueError(f'Expected canonical cutout (3,63,63), got {raw.shape}')
    encoder_type = str(runtime_config.get('image_encoder_type', '')).lower()
    if encoder_type == 'patch':
        crop = int(runtime_config['patch_size'][0])
    elif encoder_type == 'btsbot_convnext':
        crop = int(runtime_config.get('cutout_size', 63))
    else:
        raise ValueError(f'Unknown image_encoder_type: {encoder_type!r}')

    prepared = _prepare_astrominn_image(raw, runtime_config, augment=augment)
    if tuple(prepared.shape) != (3, crop, crop):
        raise RuntimeError(
            f'Upstream {encoder_type} preprocessing produced {tuple(prepared.shape)}; '
            f'expected (3,{crop},{crop}). Refusing incompatible image tokens.'
        )
    if not torch.isfinite(prepared).all():
        raise ValueError('Nonfinite image after upstream preprocessing')
    return prepared


def image_metadata_from_npz(
    path: Path, alert: int, *, runtime_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return a single historical alert stage; fail on stage/schema mismatch.

    Upstream _build_metadata_tensor is a pure instance method: it uses no self
    attributes. Calling it on None deliberately avoids copying its 55-column
    feature engineering logic, keeping parity with fusion_model_basics.
    """
    from fusion_model_basics.dataset import AlertStreamFusionDataset, _as_str_list
    with np.load(path, allow_pickle=True) as z:
        needed = {'images', 'meta_data', 'meta_columns', 'jd', 'fid'}
        if not needed.issubset(z.files):
            raise ValueError(f'New NPZ schema missing {sorted(needed-set(z.files))}: {path}')
        images = np.asarray(z['images'], dtype=np.float32)
        meta = np.asarray(z['meta_data'], dtype=np.float32)
        cols = _as_str_list(z['meta_columns'])
        jd = np.asarray(z['jd'], dtype=np.float64)
        fid = np.asarray(z['fid'], dtype=np.int16)
        provenance = np.asarray(z['provenance'], dtype=object) if 'provenance' in z.files else None
    n = len(images)
    if n == 0 or len(meta) != n or len(jd) != n or len(fid) != n:
        raise ValueError(f'New NPZ stage arrays have inconsistent lengths: {path}')
    if images.shape[1:] != (3, 63, 63):
        raise ValueError(f'Expected (N,3,63,63) images, got {images.shape}: {path}')
    if provenance is not None and len(provenance) != n:
        raise ValueError(f'Provenance length mismatch in {path}')
    alert_ids = list(range(n))
    if provenance is not None:
        for k in range(n):
            entry = provenance[k]
            if isinstance(entry, Mapping) and entry.get('alert_idx') is not None:
                val = int(entry['alert_idx'])
                if val >= 0:
                    alert_ids[k] = val
    matching = [k for k, value in enumerate(alert_ids) if value == int(alert)]
    if len(matching) != 1:
        raise ValueError(f'Historical alert {alert} matches {len(matching)} new stages in {path}')
    k = matching[0]
    # Must use upstream's own feature construction, not old 34-column preprocessing.
    metadata = AlertStreamFusionDataset._build_metadata_tensor(
        None, meta, cols, jd, fid, provenance
    )
    if metadata.shape != (n, 55):
        raise ValueError(f'Upstream metadata must have 55 features; found {tuple(metadata.shape)}')
    image = prepare_canonical_image(images[k], runtime_config=runtime_config, augment=False)
    if not torch.isfinite(metadata[k]).all():
        raise ValueError(f'Nonfinite metadata in {path}')
    return metadata[k], image, {'npz_stages': n, 'new_stage_index': k, 'alert': int(alert)}


class HistoricalNPZDataset(Dataset):
    """Adapter from historical sample-row manifests to new object NPZs.

    This requires matching object IDs and alert stage indices. It is NOT a
    fallback that silently drops unmatched rows or synthesizes 21 features.
    """
    def __init__(self, *, rows: list[dict[str, str]], index: Mapping[str, Path],
                 runtime_config: dict, object_column: str, label_column: str | None,
                 max_rows: int | None = None):
        self.rows = rows if max_rows is None else rows[:max_rows]
        self.index = index
        self.runtime_config = runtime_config
        self.object_column = object_column
        self.label_column = label_column
        for row in self.rows:
            oid = str(row[object_column])
            if oid not in index:
                raise ValueError(f'No new NPZ for historical object {oid}')
            if 'alert' not in row:
                raise ValueError('Historical manifest requires explicit alert column')
            if label_column is not None:
                label_index(row[label_column])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, k: int) -> dict:
        row = self.rows[k]
        oid = str(row[self.object_column])
        alert = int(row['alert'])
        metadata, image, extra = image_metadata_from_npz(
            self.index[oid], alert, runtime_config=self.runtime_config
        )
        item = {'obj_id': oid, 'alert': alert, 'metadata': metadata, 'image': image}
        if self.label_column is not None:
            item['target_index'] = label_index(row[self.label_column])
        return item


def manifest_dataset(path: Path, *, index: Mapping[str, Path], runtime_config: dict,
                     id_col: str, label_col: str | None) -> HistoricalNPZDataset:
    return HistoricalNPZDataset(rows=load_rows(path), index=index, runtime_config=runtime_config,
                                object_column=id_col, label_column=label_col)


def audit_npz_alert_stages(
    path: Path, alerts: set[int] | list[int] | tuple[int, ...], *, runtime_config: dict,
) -> dict[int, str | None]:
    """Audit all requested historical alerts with ONE NPZ read per object.

    This preserves the same checks as ``image_metadata_from_npz``: NPZ keys,
    aligned array lengths, canonical image shape, unique provenance alert index,
    upstream 55-column feature construction, upstream image preprocessing, and
    finite model inputs. Results are per alert (None means success); a damaged
    object produces a per-alert error without interrupting the remaining scan.

    Intended for *auditing*, not training. The regular dataset loader is kept
    unchanged, avoiding any alteration to ongoing/completed AL experiments.
    """
    requested = tuple(sorted(set(int(a) for a in alerts)))
    if not requested:
        return {}

    try:
        from fusion_model_basics.dataset import AlertStreamFusionDataset, _as_str_list

        with np.load(path, allow_pickle=True) as z:
            needed = {'images', 'meta_data', 'meta_columns', 'jd', 'fid'}
            if not needed.issubset(z.files):
                raise ValueError(f'New NPZ schema missing {sorted(needed-set(z.files))}: {path}')
            images = np.asarray(z['images'], dtype=np.float32)
            meta = np.asarray(z['meta_data'], dtype=np.float32)
            cols = _as_str_list(z['meta_columns'])
            jd = np.asarray(z['jd'], dtype=np.float64)
            fid = np.asarray(z['fid'], dtype=np.int16)
            provenance = np.asarray(z['provenance'], dtype=object) if 'provenance' in z.files else None

        n = len(images)
        if n == 0 or len(meta) != n or len(jd) != n or len(fid) != n:
            raise ValueError(f'New NPZ stage arrays have inconsistent lengths: {path}')
        if images.shape[1:] != (3, 63, 63):
            raise ValueError(f'Expected (N,3,63,63) images, got {images.shape}: {path}')
        if provenance is not None and len(provenance) != n:
            raise ValueError(f'Provenance length mismatch in {path}')

        # Exactly the same provenance lookup logic as image_metadata_from_npz.
        alert_to_indices: dict[int, list[int]] = {}
        for k in range(n):
            a = k
            if provenance is not None:
                entry = provenance[k]
                if isinstance(entry, Mapping) and entry.get('alert_idx') is not None:
                    val = int(entry['alert_idx'])
                    if val >= 0:
                        a = val
            alert_to_indices.setdefault(a, []).append(k)

        # Compute the upstream metadata transform ONCE, not once per row.
        with torch.no_grad():
            metadata = AlertStreamFusionDataset._build_metadata_tensor(
                None, meta, cols, jd, fid, provenance
            )
            if metadata.shape != (n, 55):
                raise ValueError(f'Upstream metadata must have 55 features; found {tuple(metadata.shape)}')
            outcomes: dict[int, str | None] = {}
            for alert in requested:
                try:
                    positions = alert_to_indices.get(alert, [])
                    if len(positions) != 1:
                        raise ValueError(
                            f'Historical alert {alert} matches {len(positions)} new stages in {path}'
                        )
                    k = positions[0]
                    image = prepare_canonical_image(
                        images[k], runtime_config=runtime_config, augment=False
                    )
                    if image.shape[0] != 3:
                        raise ValueError('Unexpected model inputs')
                    if not torch.isfinite(metadata[k]).all():
                        raise ValueError(f'Nonfinite metadata in {path}')
                    outcomes[alert] = None
                except Exception as exc:
                    outcomes[alert] = str(exc)
            return outcomes
    except Exception as exc:
        return {alert: str(exc) for alert in requested}
