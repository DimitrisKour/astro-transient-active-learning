"""Host-memory access to the immutable, consolidated full historical cache.

The HDD version is a network Lustre filesystem: mmap initialization is fast,
but shuffled 64-row batches cause random network page faults. Read the immutable
arrays sequentially *once per independent SLURM job*, then attach a single
shared resident ndarray to both split datasets. No extra disk copy, and no
change to sample/label ordering, model weights, or augmentation semantics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


EXPECTED_SOURCE = 'historical_full_train_plus_fixed_validation'


def attach_resident_cache(train_dataset, validation_dataset, *, cache_root: Path) -> dict:
    """Replace mapped arrays with the *same* read-only resident arrays in both splits.

    Important: callers must construct their FastHistoricalDataset instances first.
    That class still owns row/alert/label bookkeeping and the exact cache-position
    lookup. Only its array storage changes, so repeated manifest rows and hidden
    unlabeled-object semantics are unchanged.
    """
    root = Path(cache_root)
    manifest = json.loads((root/'fast_cache_manifest.json').read_text())
    if (manifest.get('format_version') != 1 or
        manifest.get('source') != EXPECTED_SOURCE or
        manifest.get('labels_saved_in_cache') is not False or
        manifest.get('test_split_accessed') is not False):
        raise ValueError('Refusing invalid, label-bearing, or test-bearing historical cache')
    n = int(manifest['unique_stages'])
    expected = (('images.npy', (n, 3, 63, 63), 'images_nbytes'),
                ('metadata.npy', (n, 55), 'metadata_nbytes'))
    print('[resident-cache] loading immutable consolidated arrays to host RAM', flush=True)
    t0 = time.perf_counter()
    arrays = {}
    file_times = {}
    for name, shape, bytes_key in expected:
        filename = root/name
        if filename.stat().st_size != int(manifest[bytes_key]):
            raise ValueError(f'Cache file length mismatch: {filename}')
        before = time.perf_counter()
        # mmap_mode=None uses ordinary sequential buffered reads. No Lustre mmap
        # page faults during shuffled training batches.
        a = np.load(filename, mmap_mode=None, allow_pickle=False)
        if a.shape != shape or a.dtype != np.dtype('float32'):
            raise ValueError(f'Invalid cache array shape/dtype: {filename}')
        if not a.flags.c_contiguous or isinstance(a, np.memmap):
            raise ValueError(f'Expected fully resident C-contiguous data: {filename}')
        a.setflags(write=False)
        arrays[name] = a
        file_times[name] = time.perf_counter() - before
        print(f'[resident-cache] {name} loaded bytes={a.nbytes} seconds={file_times[name]:.3f}', flush=True)
    im, md = arrays['images.npy'], arrays['metadata.npy']
    for ds, split in ((train_dataset,'train'), (validation_dataset,'validation')):
        if getattr(ds, 'split', None) != split:
            raise ValueError(f'Wrong dataset split: expected {split}')
        # Positions are already tied to immutable stage_index.csv. Ensure there
        # are no out-of-range indices before switching storage.
        positions = np.asarray(ds.positions, dtype=np.int64)
        if len(positions) != len(ds) or (len(positions) and
            (positions.min() < 0 or positions.max() >= n)):
            raise ValueError(f'Invalid {split} cache-position lookup')
        ds.images = im
        ds.metadata = md
    total = time.perf_counter() - t0
    report = {
        'strategy':'shared_host_resident_arrays_v3',
        'total_seconds':total, 'file_seconds':file_times,
        'resident_bytes':int(im.nbytes + md.nbytes),
        'shared_between_splits':train_dataset.images is validation_dataset.images
                                 and train_dataset.metadata is validation_dataset.metadata,
        'training_rows':len(train_dataset),
        'validation_rows':len(validation_dataset),
    }
    if not report['shared_between_splits']:
        raise AssertionError('Expected exactly one resident copy shared by both datasets')
    print('[resident-cache] ready '+json.dumps(report,sort_keys=True),flush=True)
    return report
