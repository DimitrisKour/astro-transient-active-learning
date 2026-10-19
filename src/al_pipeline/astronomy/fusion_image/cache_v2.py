"""Restartable, sharded *L0 + validation* cache for deterministic BTSbot inputs.

Each completed shard is immutable. The historical CSVs retain all label authority,
original row order, alert multiplicity, and experiment split membership. There is
no test-cache dependency for selecting a model on validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache_helpers import preprocessing_identity, read_object_stages
from .contracts import load_index, load_rows, object_labels, label_index, sha256, write_json_atomic

VERSION = 2
SPLITS = ('train', 'validation')


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _pairs(path: Path, *, id_col: str, label_col: str) -> dict[str, set[int]]:
    rows = load_rows(path)
    object_labels(rows, id_col=id_col, label_col=label_col)
    result: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        oid = str(row[id_col]).strip()
        alert = int(row['alert'])
        if alert < 0:
            raise ValueError('Negative alert stage')
        result[oid].add(alert)
    return dict(result)


def prepare(*, cache_root: Path, manifests: Mapping[str, Path], index_csv: Path,
            runtime_config: dict, id_col: str, label_col: str, n_shards: int = 24,
            upstream_sha: str | None = None) -> dict:
    if set(manifests) != set(SPLITS) or n_shards < 1:
        raise ValueError('Expected L0 train and fixed validation manifests, and positive n_shards')
    if n_shards > 128:
        raise ValueError('Too many shards')
    groups = {}
    seen = set()
    for split in SPLITS:
        by_obj = _pairs(Path(manifests[split]), id_col=id_col, label_col=label_col)
        if seen & set(by_obj):
            raise ValueError('Training and validation objects overlap')
        seen.update(by_obj)
        groups[split] = by_obj
    index = load_index(index_csv)
    missing = seen - set(index)
    if missing:
        raise ValueError(f'Missing NPZ objects: {sorted(missing)[:5]}')
    signature = preprocessing_identity(runtime_config)
    if signature['image_encoder_type'] != 'btsbot_convnext':
        raise ValueError('Only the deterministic btsbot_images preprocessing is supported')
    entries = [dict(split=split, object_id=oid, alerts=sorted(alerts))
               for split in SPLITS for oid, alerts in sorted(groups[split].items())]
    # Large objects with many eligible alerts are spread across shards as well.
    entries.sort(key=lambda e: (-len(e['alerts']), e['split'], e['object_id']))
    assignment: list[list[dict]] = [[] for _ in range(n_shards)]
    size_by_shard = [0] * n_shards
    for entry in entries:
        j = min(range(n_shards), key=lambda i: (size_by_shard[i], i))
        assignment[j].append(entry)
        size_by_shard[j] += len(entry['alerts'])
    plan = {
        'cache_version': VERSION,
        'upstream_sha': upstream_sha,
        'preprocessing_identity': signature,
        'new_index': str(Path(index_csv).resolve()),
        'new_index_sha256': sha256(index_csv),
        'id_column': id_col, 'label_column': label_col,
        'manifests': {s: str(Path(manifests[s]).resolve()) for s in SPLITS},
        'manifest_sha256': {s: sha256(manifests[s]) for s in SPLITS},
        'manifest_rows': {s: len(load_rows(manifests[s])) for s in SPLITS},
        'n_shards': n_shards, 'objects': len(entries),
        'unique_alert_stages': sum(map(len, (e['alerts'] for e in entries))),
        'assignments': assignment,
    }
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    plan_file = cache_root / 'plan.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != plan:
            raise FileExistsError('Existing cache plan differs; choose a new directory')
    else:
        if any(cache_root.iterdir()):
            raise FileExistsError('Refusing to install a cache plan into a nonempty directory')
        write_json_atomic(plan_file, plan)
    print(f'[cache-v2] planned {len(entries)} objects, '
          f'{plan["unique_alert_stages"]} unique alert stages, {n_shards} shards', flush=True)
    return plan


def _plan(root: Path) -> dict:
    plan = json.loads((root / 'plan.json').read_text())
    if plan.get('cache_version') != VERSION:
        raise ValueError('Invalid cache version')
    for name, path in plan['manifests'].items():
        if sha256(path) != plan['manifest_sha256'][name]:
            raise ValueError(f'Changed historical {name} manifest')
    if sha256(plan['new_index']) != plan['new_index_sha256']:
        raise ValueError('Changed indexed NPZ dataset')
    return plan


def build_shard(*, root: Path, shard_id: int, runtime_config: dict,
                verify_n: int = 2, progress_every: int = 10) -> dict:
    root = Path(root)
    plan = _plan(root)
    if preprocessing_identity(runtime_config) != plan['preprocessing_identity']:
        raise ValueError('Current upstream preprocessing differs from cache plan')
    if not 0 <= shard_id < plan['n_shards']:
        raise ValueError('Shard ID out of bounds')
    torch.set_num_threads(1)  # prevent 1-core SLURM allocations spawning many torch threads
    out = root / f'shard_{shard_id:03d}'
    assignments = plan['assignments'][shard_id]
    if not assignments:
        raise ValueError('Empty cache shard; reduce shard count')
    expected = sum(len(e['alerts']) for e in assignments)
    if (out / 'done.json').exists():
        done = json.loads((out / 'done.json').read_text())
        if done.get('stages') != expected or done.get('plan_sha256') != sha256(root / 'plan.json'):
            raise ValueError(f'Mismatched previously completed cache shard: {out}')
        for name in ('images.npy', 'metadata.npy', 'stage_index.csv'):
            if not (out / name).is_file():
                raise FileNotFoundError(out / name)
        print(f'[cache-v2] skipping complete shard {shard_id}', flush=True)
        return done
    if out.exists():
        raise FileExistsError(f'Incomplete shard directory found: {out}; inspect first')
    # SHARD-SCOPED temp files: timeout affects only this shard, not completed shards.
    work = Path(tempfile.mkdtemp(prefix=f'.shard_{shard_id:03d}_', dir=root))
    started = time.perf_counter()
    progress_path = root / f'progress_{shard_id:03d}.jsonl'
    index = load_index(plan['new_index'])
    records = []
    next_row = 0
    images = np.lib.format.open_memmap(work/'images.npy', mode='w+',
        dtype=np.float32, shape=(expected, 3,63,63))
    metadata = np.lib.format.open_memmap(work/'metadata.npy', mode='w+',
        dtype=np.float32, shape=(expected,55))
    try:
        with progress_path.open('a', encoding='utf-8', buffering=1) as progress:
            for j, entry in enumerate(assignments):
                oid = entry['object_id']
                t0 = time.perf_counter()
                result = read_object_stages(index[oid], set(entry['alerts']), runtime_config)
                for alert in entry['alerts']:
                    md, im = result[alert]
                    if md.shape != (55,) or im.shape != (3,63,63):
                        raise ValueError(f'Invalid prepared arrays for {oid}/{alert}')
                    metadata[next_row] = md
                    images[next_row] = im
                    records.append((entry['split'], oid, int(alert), next_row))
                    next_row += 1
                elapsed = time.perf_counter() - t0
                progress.write(json.dumps({'object': oid, 'split':entry['split'],
                    'number':j+1, 'total':len(assignments), 'npz_bytes':index[oid].stat().st_size,
                    'object_seconds':round(elapsed,5)}, sort_keys=True)+'\n')
                if (j+1) % progress_every == 0 or j+1 == len(assignments):
                    print(f'[cache-v2] shard {shard_id:02d} {j+1}/{len(assignments)} objects '
                          f'{next_row}/{expected} stages, '
                          f'{time.perf_counter()-started:.1f}s',flush=True)
        if next_row != expected:
            raise RuntimeError('Incomplete cache shard')
        images.flush(); metadata.flush(); del images, metadata
        with (work/'stage_index.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('split','object_id','alert','position'))
            writer.writerows(records)
        # Numerically compare against independent ORIGINAL per-alert loader.
        from .data import image_metadata_from_npz
        imgs = np.load(work/'images.npy',mmap_mode='r')
        mds = np.load(work/'metadata.npy',mmap_mode='r')
        for k in np.linspace(0, expected-1, min(expected,verify_n), dtype=int):
            split, oid, alert, pos = records[int(k)]
            base_md, base_img, _ = image_metadata_from_npz(
                index[oid],alert,runtime_config=runtime_config)
            if not np.array_equal(np.asarray(base_md),np.asarray(mds[pos])) or \
               not np.array_equal(np.asarray(base_img),np.asarray(imgs[pos])):
                raise ValueError(f'Cache-v2 numerical mismatch {split}:{oid}:{alert}')
        del imgs, mds
        done = {'shard':shard_id,'objects':len(assignments),'stages':next_row,
                'elapsed_seconds':time.perf_counter()-started,
                'plan_sha256':sha256(root/'plan.json'),
                'index_sha256':sha256(work/'stage_index.csv'),
                'numeric_checks':min(expected,verify_n)}
        write_json_atomic(work/'done.json',done)
        work.rename(out)
        return done
    except BaseException:
        # Preserve progress JSONL even on exception; temp arrays remain for diagnostics.
        print(f'[cache-v2] shard {shard_id} failed; inspect {work}', flush=True)
        raise


def finalize(root: Path, *, source: str = 'L0+validation only') -> dict:
    root = Path(root)
    plan = _plan(root)
    seen = set()
    total_objects = 0
    total_stages = 0
    summary = {}
    for j, entries in enumerate(plan['assignments']):
        sroot = root/f'shard_{j:03d}'
        done_file = sroot/'done.json'
        if not done_file.is_file():
            raise FileNotFoundError(f'Incomplete shard {j}: {done_file}')
        done = json.loads(done_file.read_text())
        if done['plan_sha256'] != sha256(root/'plan.json') or done['stages'] != sum(len(e['alerts']) for e in entries):
            raise ValueError(f'Shard {j} identity mismatch')
        if sha256(sroot/'stage_index.csv') != done['index_sha256']:
            raise ValueError(f'Shard {j} stage index modified')
        with (sroot/'stage_index.csv').open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        if len(rows) != done['stages']:
            raise ValueError(f'Shard {j} row-count mismatch')
        for row in rows:
            key = row['split'],row['object_id'],int(row['alert'])
            if key in seen:
                raise ValueError(f'Duplicate cache stage {key}')
            seen.add(key)
        if np.load(sroot/'images.npy',mmap_mode='r').shape != (done['stages'],3,63,63):
            raise ValueError(f'Shard {j} image array shape mismatch')
        if np.load(sroot/'metadata.npy',mmap_mode='r').shape != (done['stages'],55):
            raise ValueError(f'Shard {j} metadata array shape mismatch')
        total_objects += done['objects']
        total_stages += done['stages']
        summary[str(j)] = {'objects':done['objects'],'stages':done['stages'],
                           'seconds':done['elapsed_seconds']}
    if total_objects != plan['objects'] or total_stages != plan['unique_alert_stages']:
        raise ValueError('Incomplete combined cache')
    payload = {'cache_version': VERSION,
        'preprocessing_identity':plan['preprocessing_identity'],
        'new_index_sha256':plan['new_index_sha256'],
        'manifest_sha256':plan['manifest_sha256'],
        'manifest_rows':plan['manifest_rows'],
        'plan_sha256':sha256(root/'plan.json'),
        'unique_objects':total_objects, 'unique_stages':total_stages,
        'shards':summary, 'source':source,
        'labels_saved_in_cache':False}
    final_path = root/'cache_manifest.json'
    if final_path.exists() and json.loads(final_path.read_text()) != payload:
        raise FileExistsError('Different completed cache manifest already exists')
    write_json_atomic(final_path,payload)
    print(f'[cache-v2] FINALIZED {total_objects} objects, {total_stages} unique stages',flush=True)
    return payload


class ShardedHistoricalNPZDataset(Dataset):
    """Reusable sharded-cache dataset preserving every original historical CSV row."""
    def __init__(self, *, rows: list[dict[str,str]], cache_root:Path,
                 runtime_config:dict, object_column:str,label_column:str|None,
                 preload:bool=True):
        self.rows=rows
        root=Path(cache_root)
        meta=json.loads((root/'cache_manifest.json').read_text())
        if meta['cache_version'] != VERSION or meta['preprocessing_identity'] != preprocessing_identity(runtime_config):
            raise ValueError('Sharded-cache preprocessing mismatch')
        if sha256(root/'plan.json') != meta['plan_sha256']:
            raise ValueError('Sharded-cache plan modified')
        lookup={}
        self.images={}
        self.metadata={}
        for s in sorted(meta['shards'],key=int):
            shard=root/f'shard_{int(s):03d}'
            with (shard/'stage_index.csv').open(newline='',encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    key=r['object_id'],int(r['alert'])
                    if key in lookup:
                        raise ValueError(f'Duplicate cache stage {key}')
                    lookup[key]=(s,int(r['position']))
            self.images[s]=np.load(shard/'images.npy',mmap_mode='r')
            self.metadata[s]=np.load(shard/'metadata.npy',mmap_mode='r')
        self.refs=[]
        for r in rows:
            key=str(r[object_column]),int(r['alert'])
            if key not in lookup:
                raise ValueError(f'Stage not in L0/validation cache: {key}')
            self.refs.append(lookup[key])
            if label_column is not None:
                label_index(r[label_column])
        if preload:
            # One contiguous array per dataset avoids per-batch disk accesses.
            ims=np.empty((len(rows),3,63,63),dtype=np.float32)
            mds=np.empty((len(rows),55),dtype=np.float32)
            # Gather per shard instead of 17k interleaved HDD seeks.
            by_shard=defaultdict(list)
            for i,(s,pos) in enumerate(self.refs):
                by_shard[s].append((i,pos))
            for s, positions in by_shard.items():
                destination=np.asarray([i for i,_ in positions],dtype=np.int64)
                source=np.asarray([p for _,p in positions],dtype=np.int64)
                ims[destination]=self.images[s][source]
                mds[destination]=self.metadata[s][source]
            self.images=ims
            self.metadata=mds
        self.preloaded=preload
        self.object_column=object_column
        self.label_column=label_column
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i:int):
        r=self.rows[i]
        if self.preloaded:
            img=self.images[i].copy()
            md=self.metadata[i].copy()
        else:
            s,p=self.refs[i]
            img=np.asarray(self.images[s][p]).copy()
            md=np.asarray(self.metadata[s][p]).copy()
        item={'obj_id':str(r[self.object_column]),'alert':int(r['alert']),
              'image':torch.from_numpy(img),'metadata':torch.from_numpy(md)}
        if self.label_column is not None:
            item['target_index']=label_index(r[self.label_column])
        return item
