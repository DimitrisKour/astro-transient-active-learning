"""Immutable, LABEL-FREE, independent historical TEST stage cache for ConvNeXt.

Unlike frozen Task 0 train/validation caches, this module deliberately handles
only the test split. Preprocessing can run before final-test evaluation; neither
labels nor metrics are stored. Test predictions require a separately frozen audit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache_helpers import preprocessing_identity, read_object_stages
from .contracts import load_index, load_rows, sha256, write_json_atomic, label_index

VERSION = 1
SOURCE = "standalone_astrominn_historical_test_only"
DEFAULT_ROOT = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_v1')
DEFAULT_TASK0 = Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task0_full_historical_v1')
DEFAULT_SOURCE = Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_matrix_v1/plan.json')
DEFAULT_TRAIN_CACHE = Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_full_historical_fastcache_hdd_v1')


def _stages(rows: list[dict], id_col: str) -> tuple[dict[str, list[int]], int]:
    by_object = defaultdict(set)
    for r in rows:
        oid = str(r[id_col]).strip()
        if not oid:
            raise ValueError('Empty object ID')
        alert = int(r['alert'])
        if alert < 0:
            raise ValueError('Negative alert stage')
        by_object[oid].add(alert)
    return {oid: sorted(stages) for oid, stages in sorted(by_object.items())}, sum(map(len, by_object.values()))


def _manifest_stages_only(path: Path, id_col: str) -> tuple[dict[str, list[int]], int]:
    """Never pull test label values into the cache-preparation code path."""
    groups=defaultdict(set)
    count=0
    with Path(path).open(newline='',encoding='utf-8') as f:
        rd=csv.DictReader(f)
        if rd.fieldnames is None or not {id_col,'alert'}.issubset(rd.fieldnames):
            raise ValueError(f'Missing object/alert columns in {path}')
        for r in rd:
            oid=str(r[id_col]).strip()
            alert=int(r['alert'])
            if not oid or alert<0:
                raise ValueError('Invalid TEST object/alert')
            groups[oid].add(alert)
            count+=1
    if count==0:
        raise ValueError(f'Empty historical manifest {path}')
    return {k:sorted(v) for k,v in sorted(groups.items())}, count


def plan_inputs(*, task0: Path, source_plan: Path, training_cache: Path) -> tuple[dict, dict, dict]:
    """Read provenance and STAGE IDS only; never parse or store test targets."""
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from fusion_model_basics.config import build_astrominn_config
    from .contracts import class_order_config

    frozen = json.loads((task0/'plan.json').read_text())
    if frozen.get('experiment') != 'task0_full_history_btsbot_pretrained_vs_randomized':
        raise ValueError('Not the frozen complete-history Task 0 plan')
    if sha256(source_plan) != frozen['source_plan_sha256']:
        raise ValueError('Changed original Task 0 source plan')
    source = json.loads(source_plan.read_text())
    registry = DatasetRegistry.from_toml(source['registry']).resolve_profile('astrominn_ztf').dataset
    train = Path(registry.train_manifest)
    val = Path(registry.validation_manifest)
    test = Path(registry.test_manifest)
    if not test.is_file():
        raise FileNotFoundError('Missing fixed historical test manifest')
    if sha256(train) != frozen['train_manifest_sha256'] or sha256(val) != frozen['validation_manifest_sha256']:
        raise ValueError('Task 0 train/validation manifests differ from frozen plan')
    if sha256(source['new_index']) != source['file_hashes']['new_index']:
        raise ValueError('Original per-object NPZ index changed')
    old_fast = json.loads((training_cache/'fast_cache_manifest.json').read_text())
    if old_fast.get('source') != 'historical_full_train_plus_fixed_validation' or old_fast.get('labels_saved_in_cache') is not False or old_fast.get('test_split_accessed') is not False:
        raise ValueError('Old train/validation cache identity incorrect')
    if sha256(training_cache/'fast_cache_manifest.json') != frozen['cache_manifest_sha256']:
        raise ValueError('Frozen Task 0 train/validation cache has changed')
    cfg = build_astrominn_config('btsbot_images',coord_checkpoint=None,
          hf_image_model_dir=source['hf_snapshot'],classes=class_order_config(),
          device='cpu',dir='.')
    signature = preprocessing_identity(cfg)
    if signature != old_fast['preprocessing_identity']:
        raise ValueError('Test preprocessing differs from full-historical training cache')
    id_col = registry.object_id_column
    label_col = registry.label_column
    split_paths = {'train':train, 'validation':val, 'test':test}
    stage_counts={k:_manifest_stages_only(v,id_col) for k,v in split_paths.items()}
    stages_by_split={k:v[0] for k,v in stage_counts.items()}
    for x,y in (('train','validation'),('train','test'),('validation','test')):
        if set(stages_by_split[x]) & set(stages_by_split[y]):
            raise ValueError(f'Historical object overlap: {x}/{y}')
    test_groups=stages_by_split['test']
    index=load_index(source['new_index'])
    if set(test_groups)-set(index):
        raise ValueError('Original NPZ index missing TEST objects')
    return ({
        'test_manifest':str(test.resolve()),'test_manifest_sha256':sha256(test),
        'test_rows':stage_counts['test'][1], 'test_objects':len(test_groups),
        'test_unique_stages':sum(len(v) for v in test_groups.values()),
        'train_manifest_sha256':sha256(train),
        'validation_manifest_sha256':sha256(val),
        'test_id_column':id_col, 'test_label_column':label_col,
        'new_index':str(Path(source['new_index']).resolve()),
        'new_index_sha256':sha256(source['new_index']),
        'source_plan_sha256':sha256(source_plan),
        'task0_plan_sha256':sha256(task0/'plan.json'),
        'train_cache_manifest_sha256':sha256(training_cache/'fast_cache_manifest.json'),
        'preprocessing_identity':signature,
    }, test_groups, cfg)


def prepare(*, root: Path=DEFAULT_ROOT, task0: Path=DEFAULT_TASK0,
            source_plan: Path=DEFAULT_SOURCE, training_cache: Path=DEFAULT_TRAIN_CACHE,
            n_shards: int=24) -> dict:
    if not 1 <= n_shards <= 64:
        raise ValueError('Shard count must be in [1,64]')
    info, by_object, _ = plan_inputs(task0=task0,source_plan=source_plan,training_cache=training_cache)
    # Greedy balancing by STAGES; one object/NPZ read per shard.
    objects=sorted(by_object.items(),key=lambda entry:(-len(entry[1]),entry[0]))
    if n_shards > len(objects):
        raise ValueError('More shards than test objects')
    allocations=[[] for _ in range(n_shards)];sizes=[0]*n_shards
    for oid,alerts in objects:
        j=min(range(n_shards),key=lambda k:(sizes[k],k))
        allocations[j].append({'object_id':oid,'alerts':alerts})
        sizes[j]+=len(alerts)
    plan={**info,'format_version':VERSION,'source':SOURCE,
          'n_shards':n_shards,'assignments':allocations,'labels_saved_in_cache':False,
          'predictions_saved_in_cache':False,
          'cache_module_sha256':sha256(__file__),
          'plan_created_for':'CPU-only precomputation, without labels or metrics'}
    root.mkdir(parents=True,exist_ok=True)
    target=root/'plan.json'
    if target.exists():
        if json.loads(target.read_text())!=plan:
            raise ValueError('Frozen TEST cache plan changed: choose a new root')
    else:
        if any(root.iterdir()):
            raise FileExistsError('Refusing preparation in nonempty TEST cache directory')
        write_json_atomic(target,plan)
    return {'prepared':True,'cache_root':str(root),'n_shards':n_shards,
            'test_objects':info['test_objects'],'test_rows':info['test_rows'],
            'unique_stages':info['test_unique_stages'],'labels_stored':False}


def frozen_plan(root: Path, *, check_module: bool=True) -> dict:
    root=Path(root)
    plan=json.loads((root/'plan.json').read_text())
    if plan.get('source') != SOURCE or plan.get('format_version') != VERSION:
        raise ValueError('Not a dedicated immutable TEST cache')
    if check_module and plan.get('cache_module_sha256') != sha256(__file__):
        raise ValueError('TEST cache builder changed after freezing plan')
    for fname, key in (
        (plan['test_manifest'],'test_manifest_sha256'),
        (plan['new_index'],'new_index_sha256')):
        if sha256(fname) != plan[key]:
            raise ValueError(f'Source file changed after TEST cache plan: {fname}')
    return plan


def build_shard(root: Path, shard_id: int, runtime_config: dict, *, verify_samples:int=2) -> dict:
    plan=frozen_plan(root)
    if preprocessing_identity(runtime_config) != plan['preprocessing_identity']:
        raise ValueError('Cache preprocessing mismatch')
    if not 0 <= shard_id < plan['n_shards']:
        raise ValueError('Shard number out of range')
    group=plan['assignments'][shard_id]
    if not group:
        raise ValueError('Empty assigned shard')
    output=Path(root)/f'shard_{shard_id:03d}'
    done_path=output/'done.json'
    if done_path.is_file():
        done=json.loads(done_path.read_text())
        if done.get('plan_sha256')!=sha256(Path(root)/'plan.json'):
            raise ValueError('Existing shard has different plan')
        for key in ('images.npy','metadata.npy','stage_index.csv'):
            if not (output/key).is_file():
                raise FileNotFoundError(output/key)
        if done.get('index_sha256') != sha256(output/'stage_index.csv'):
            raise ValueError('Existing shard index has changed')
        return done
    if output.exists():
        raise FileExistsError('Found incomplete shard; inspect before retrying')
    tmp=Path(tempfile.mkdtemp(prefix=f'.shard_{shard_id:03d}_',dir=root))
    n=sum(len(x['alerts']) for x in group)
    started=time.perf_counter()
    try:
        # 24-way shards typically need only tens of MB of resident memory.
        image=np.empty((n,3,63,63),dtype=np.float32)
        metadata=np.empty((n,55),dtype=np.float32)
        records=[];pos=0
        index=load_index(plan['new_index'])
        torch.set_num_threads(1)
        for j,entry in enumerate(group):
            oid=entry['object_id']
            values=read_object_stages(index[oid], set(entry['alerts']), runtime_config)
            if set(values)!=set(entry['alerts']):
                raise ValueError(f'Incomplete NPZ stages for {oid}')
            for alert in entry['alerts']:
                md,im=values[alert]
                md=np.asarray(md,dtype=np.float32)
                im=np.asarray(im,dtype=np.float32)
                if md.shape!=(55,) or im.shape!=(3,63,63) or not np.isfinite(md).all() or not np.isfinite(im).all():
                    raise ValueError(f'Invalid model inputs for {oid}/{alert}')
                image[pos]=im;metadata[pos]=md
                records.append(('test',oid,int(alert),pos))
                pos+=1
            if (j+1)%25==0 or j+1==len(group):
                print(f'[test-cache] shard {shard_id} objects {j+1}/{len(group)} stages {pos}/{n} seconds {time.perf_counter()-started:.1f}',flush=True)
        assert pos==n
        with (tmp/'images.npy').open('wb') as f:np.save(f,image,allow_pickle=False)
        with (tmp/'metadata.npy').open('wb') as f:np.save(f,metadata,allow_pickle=False)
        with (tmp/'stage_index.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(('split','object_id','alert','position'));w.writerows(records)
        # Two independently loaded reference stages per shard; preprocessing parity.
        from .data import image_metadata_from_npz
        for k in np.linspace(0,n-1,min(n,verify_samples),dtype=int):
            _,oid,alert,p=records[int(k)]
            m2,i2,_=image_metadata_from_npz(index[oid],alert,runtime_config=runtime_config)
            if not np.array_equal(metadata[p],np.asarray(m2)) or not np.array_equal(image[p],np.asarray(i2)):
                raise ValueError(f'Independent NPZ numerical parity failed: {oid}/{alert}')
        summary={'shard':shard_id,'stages':n,'objects':len(group),
                 'plan_sha256':sha256(Path(root)/'plan.json'),
                 'index_sha256':sha256(tmp/'stage_index.csv'),
                 'images_nbytes':(tmp/'images.npy').stat().st_size,
                 'metadata_nbytes':(tmp/'metadata.npy').stat().st_size,
                 'checks':min(n,verify_samples),'seconds':time.perf_counter()-started}
        write_json_atomic(tmp/'done.json',summary)
        tmp.rename(output)
        return summary
    except BaseException:
        print(f'[test-cache] interrupted shard {shard_id}: partial temp {tmp}',flush=True)
        raise


def consolidate(*, root: Path, target: Path) -> dict:
    """Single sequential buffered copy; never mmap-write to Lustre or NVMe."""
    root=Path(root);target=Path(target)
    plan=frozen_plan(root)
    if target.exists():
        raise FileExistsError(f'Final TEST cache already exists: {target}')
    n=plan['test_unique_stages']
    total_bytes=n*(3*63*63+55)*4
    target.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(target.parent).free < total_bytes + 128*1024*1024:
        raise OSError('Insufficient apparent filesystem capacity for consolidated TEST cache')
    tmp=Path(tempfile.mkdtemp(prefix=f'.{target.name}.building_',dir=target.parent))
    img_shape=(n,3,63,63); md_shape=(n,55)
    written=0;seen=set();positions=set();source_checks=0
    try:
        with (tmp/'images.npy').open('wb',buffering=1024*1024) as fi, \
             (tmp/'metadata.npy').open('wb',buffering=1024*1024) as fm, \
             (tmp/'stage_index.csv').open('w',newline='',encoding='utf-8') as ix:
            dtype=np.dtype('<f4')
            for dst,shape in ((fi,img_shape),(fm,md_shape)):
                np.lib.format.write_array_header_2_0(dst,{
                    'descr':np.lib.format.dtype_to_descr(dtype),
                    'fortran_order':False,'shape':shape})
            writer=csv.writer(ix);writer.writerow(('split','object_id','alert','position'))
            for shard_id, entries in enumerate(plan['assignments']):
                base=root/f'shard_{shard_id:03d}'
                done=json.loads((base/'done.json').read_text())
                if done['plan_sha256']!=sha256(root/'plan.json') or done['stages']!=sum(len(q['alerts']) for q in entries):
                    raise ValueError(f'Untrusted test shard {shard_id}')
                if done['index_sha256']!=sha256(base/'stage_index.csv'):
                    raise ValueError(f'Test shard index mismatch {shard_id}')
                with (base/'images.npy').open('rb') as src:image=np.load(src,allow_pickle=False)
                with (base/'metadata.npy').open('rb') as src:metadata=np.load(src,allow_pickle=False)
                count=done['stages']
                if image.shape!=(count,3,63,63) or metadata.shape!=(count,55) or image.dtype!=dtype or metadata.dtype!=dtype:
                    raise ValueError('Unexpected shard schema')
                with (base/'stage_index.csv').open(newline='',encoding='utf-8') as sf:rows=list(csv.DictReader(sf))
                if len(rows)!=count or sorted(int(r['position']) for r in rows)!=list(range(count)):
                    raise ValueError(f'Incomplete shard index {shard_id}')
                for r in rows:
                    key=(r['object_id'],int(r['alert']))
                    if r['split']!='test' or key in seen:
                        raise ValueError('Incorrect split or duplicate test stage')
                    seen.add(key)
                    writer.writerow(('test',r['object_id'],int(r['alert']),written+int(r['position'])))
                # A single buffered write for each shard: low I/O syscall overhead.
                fi.write(memoryview(np.ascontiguousarray(image)).cast('B'))
                fm.write(memoryview(np.ascontiguousarray(metadata)).cast('B'))
                written+=count
                source_checks+=done['checks']
                print(f'[test-cache] consolidated shard {shard_id+1}/{plan["n_shards"]} stages={written}/{n}',flush=True)
            for f in (fi,fm,ix): f.flush();os.fsync(f.fileno())
        expected={(e['object_id'],int(alert)) for shard in plan['assignments'] for e in shard for alert in e['alerts']}
        if written!=n or len(seen)!=n or seen!=expected:
            raise ValueError('Consolidated stage coverage mismatch')
        manifest={k:plan[k] for k in (
            'test_manifest','test_manifest_sha256','test_rows','test_objects','test_unique_stages',
            'train_manifest_sha256','validation_manifest_sha256','test_id_column',
            'test_label_column','new_index_sha256','source_plan_sha256','task0_plan_sha256',
            'train_cache_manifest_sha256','preprocessing_identity')}
        manifest.update({'format_version':VERSION,'source':SOURCE,'labels_saved_in_cache':False,
            'predictions_saved_in_cache':False,'plan_sha256':sha256(root/'plan.json'),
            'shards_root':str(root.resolve()),'shard_numerical_checks':source_checks,
            'stage_index_sha256':sha256(tmp/'stage_index.csv'),
            'images_nbytes':(tmp/'images.npy').stat().st_size,
            'metadata_nbytes':(tmp/'metadata.npy').stat().st_size,
            'images_sha256':sha256(tmp/'images.npy'),
            'metadata_sha256':sha256(tmp/'metadata.npy'),
            'writer':'buffered_sequential_npys_v1'})
        write_json_atomic(tmp/'fast_cache_manifest.json',manifest)
        validate(root=tmp,manifest_path=plan['test_manifest'],expected=manifest,
                 verify_shards=2)
        tmp.rename(target)
        return manifest
    except BaseException:
        print(f'[test-cache] consolidation failed; partial data retained {tmp}',flush=True)
        raise


def validate(*, root: Path, manifest_path: Path | None=None,
             expected: dict | None=None, verify_shards: int=2) -> dict:
    root=Path(root)
    meta=json.loads((root/'fast_cache_manifest.json').read_text())
    if meta.get('source')!=SOURCE or meta.get('format_version')!=VERSION or meta.get('labels_saved_in_cache') is not False or meta.get('predictions_saved_in_cache') is not False:
        raise ValueError('Not a dedicated label-free TEST cache')
    for fname,key in (('images.npy','images_nbytes'),('metadata.npy','metadata_nbytes')):
        if (root/fname).stat().st_size!=meta[key]:
            raise ValueError(f'Incorrect {fname} byte size')
        if sha256(root/fname)!=meta[fname.replace('.npy','_sha256')]:
            raise ValueError(f'Modified immutable TEST array: {fname}')
    if sha256(root/'stage_index.csv')!=meta['stage_index_sha256']:
        raise ValueError('TEST cache stage index modified')
    path=Path(manifest_path or meta['test_manifest'])
    if sha256(path)!=meta['test_manifest_sha256']:
        raise ValueError('TEST input manifest changed')
    imgs=np.load(root/'images.npy',mmap_mode='r',allow_pickle=False)
    md=np.load(root/'metadata.npy',mmap_mode='r',allow_pickle=False)
    n=meta['test_unique_stages']
    if imgs.shape!=(n,3,63,63) or md.shape!=(n,55) or imgs.dtype!=np.float32 or md.dtype!=np.float32:
        raise ValueError('Consolidated test tensors wrong shape/dtype')
    keys={};used=set()
    with (root/'stage_index.csv').open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key=(row['object_id'],int(row['alert']))
            i=int(row['position'])
            if row['split']!='test' or key in keys or i in used or not 0<=i<n:
                raise ValueError('Invalid immutable test-cache stage index')
            keys[key]=i;used.add(i)
    if len(keys)!=n or used!=set(range(n)):
        raise ValueError('Incomplete test-cache stage coverage')
    expected_keys={(str(r[meta['test_id_column']]),int(r['alert'])) for r in load_rows(path)}
    if expected_keys!=set(keys) or len(load_rows(path))!=meta['test_rows']:
        raise ValueError('Test manifest stage coverage mismatch')
    if expected and expected['stage_index_sha256']!=meta['stage_index_sha256']:
        raise ValueError('Wrong consolidated test source')
    verified=0
    if verify_shards:
        shards_root=Path(meta['shards_root'])
        frozen=frozen_plan(shards_root)
        if sha256(shards_root/'plan.json')!=meta['plan_sha256']:
            raise ValueError('Frozen source shard plan changed')
        for shard_id in range(frozen['n_shards']):
            d=shards_root/f'shard_{shard_id:03d}'
            with (d/'stage_index.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            im=np.load(d/'images.npy',mmap_mode='r',allow_pickle=False)
            mm=np.load(d/'metadata.npy',mmap_mode='r',allow_pickle=False)
            for k in np.linspace(0,len(rows)-1,min(verify_shards,len(rows)),dtype=int):
                r=rows[int(k)]; pos=keys[r['object_id'],int(r['alert'])]
                if not np.array_equal(imgs[pos],im[int(r['position'])]) or not np.array_equal(md[pos],mm[int(r['position'])]):
                    raise ValueError('Test cache numerical comparison failed')
                verified+=1
    return {'validated':True,'test_rows':meta['test_rows'],'objects':meta['test_objects'],
            'unique_stages':n,'shard_spot_checks':verified,
            'cache_bytes':int(imgs.nbytes+md.nbytes),'labels_present':False}


class TestFastDataset(Dataset):
    """Keep original manifest row order/multiplicity; stage index has no labels."""
    def __init__(self, *, rows:list[dict],root:Path,id_col:str,label_col:str|None):
        root=Path(root)
        m=json.loads((root/'fast_cache_manifest.json').read_text())
        if m.get('source')!=SOURCE or m.get('labels_saved_in_cache') is not False:
            raise ValueError('Invalid label-free TEST cache')
        self.images=np.load(root/'images.npy',mmap_mode='r',allow_pickle=False)
        self.metadata=np.load(root/'metadata.npy',mmap_mode='r',allow_pickle=False)
        if self.images.shape!=(m['test_unique_stages'],3,63,63) or self.metadata.shape!=(m['test_unique_stages'],55):
            raise ValueError('Invalid cached TEST array schema')
        key2pos={}
        with (root/'stage_index.csv').open(newline='',encoding='utf-8') as f:
            for record in csv.DictReader(f):
                key=(record['object_id'],int(record['alert']))
                if record['split']!='test' or key in key2pos:
                    raise ValueError('Invalid test stage index')
                key2pos[key]=int(record['position'])
        self.positions=[];self.rows=rows;self.id_col=id_col;self.label_col=label_col
        for row in rows:
            key=(str(row[id_col]),int(row['alert']))
            if key not in key2pos:
                raise KeyError(f'Missing test cache stage: {key}')
            self.positions.append(key2pos[key])
            if label_col is not None:label_index(row[label_col])

    def __len__(self):return len(self.positions)

    def __getitem__(self,i):
        row=self.rows[i];p=self.positions[i]
        item={'obj_id':str(row[self.id_col]),'alert':int(row['alert']),
              'image':torch.from_numpy(np.asarray(self.images[p]).copy()),
              'metadata':torch.from_numpy(np.asarray(self.metadata[p]).copy())}
        if self.label_col is not None:item['target_index']=label_index(row[self.label_col])
        return item


def attach_resident(dataset:TestFastDataset, *, root:Path,verify_files:bool=True) -> dict:
    """One buffered sequential load per file; avoid random Lustre page faults."""
    if verify_files:
        validate(root=root,verify_shards=0)
    m=json.loads((Path(root)/'fast_cache_manifest.json').read_text())
    started=time.perf_counter()
    img=np.load(Path(root)/'images.npy',allow_pickle=False)
    md=np.load(Path(root)/'metadata.npy',allow_pickle=False)
    if img.shape!=(m['test_unique_stages'],3,63,63) or md.shape!=(m['test_unique_stages'],55) or img.dtype!=np.float32 or md.dtype!=np.float32:
        raise ValueError('Unexpected TEST resident array shape')
    if isinstance(img,np.memmap) or isinstance(md,np.memmap):
        raise ValueError('Resident array unexpectedly still memory mapped')
    img.setflags(write=False);md.setflags(write=False)
    dataset.images=img;dataset.metadata=md
    return {'strategy':'single_host_resident_buffered_load_v1',
            'load_seconds':time.perf_counter()-started,
            'resident_bytes':int(img.nbytes+md.nbytes),
            'array_read_only':not img.flags.writeable and not md.flags.writeable}
