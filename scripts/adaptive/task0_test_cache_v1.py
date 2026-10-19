#!/usr/bin/env python3
"""CPU-only, label-free, reproducible test-cache plan/shard/consolidate/verify."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

from al_pipeline.astronomy.fusion_image.test_fastcache_v1 import (
    DEFAULT_ROOT,DEFAULT_SOURCE,DEFAULT_TASK0,DEFAULT_TRAIN_CACHE,
    prepare,plan_inputs,build_shard,consolidate,validate,frozen_plan,
)
from al_pipeline.astronomy.fusion_image.cache_helpers import preprocessing_identity
from al_pipeline.astronomy.fusion_image.contracts import class_order_config,sha256

SHARDS=Path(str(DEFAULT_ROOT)+'_shards')


def runtime(source_plan):
    from fusion_model_basics.config import build_astrominn_config
    original=json.loads(source_plan.read_text())
    rev=subprocess.check_output(['git','-C',original['upstream_repo'],'rev-parse','HEAD'],text=True).strip()
    if rev!=original['model_sha']:
        raise ValueError('Upstream model revision changed')
    if sha256(Path(original['hf_snapshot'])/'pytorch_model.bin')!=original['hf_weights_sha256']:
        raise ValueError('HF model weights changed')
    return build_astrominn_config('btsbot_images',coord_checkpoint=None,
        hf_image_model_dir=original['hf_snapshot'],classes=class_order_config(),
        device='cpu',dir='.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','prepare','shard','consolidate','validate','status','probe'))
    p.add_argument('--task0',type=Path,default=DEFAULT_TASK0)
    p.add_argument('--source-plan',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--training-cache',type=Path,default=DEFAULT_TRAIN_CACHE)
    p.add_argument('--shards-root',type=Path,default=SHARDS)
    p.add_argument('--target',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--n-shards',type=int,default=24)
    p.add_argument('--shard-id',type=int)
    args=p.parse_args()
    if args.command=='check':
        info,groups,cfg=plan_inputs(task0=args.task0,source_plan=args.source_plan,
                                   training_cache=args.training_cache)
        runtime_cfg=runtime(args.source_plan)
        if preprocessing_identity(cfg)!=preprocessing_identity(runtime_cfg):
            raise ValueError('Test config reproducibility mismatch')
        result={'check':'passed','test_manifest_sha256':info['test_manifest_sha256'],
            'test_objects':info['test_objects'],'test_rows':info['test_rows'],
            'test_unique_stages':info['test_unique_stages'],
            'train_validation_test_disjoint':True,
            'same_preprocessing_as_task0':True,'test_labels_loaded_for_cache':False,
            'test_predictions_computed':False}
    elif args.command=='prepare':
        runtime(args.source_plan)
        result=prepare(root=args.shards_root,task0=args.task0,source_plan=args.source_plan,
                       training_cache=args.training_cache,n_shards=args.n_shards)
    elif args.command=='shard':
        if args.shard_id is None:p.error('shard requires --shard-id')
        cfg=runtime(args.source_plan)
        result=build_shard(args.shards_root,args.shard_id,cfg)
    elif args.command=='consolidate':
        result=consolidate(root=args.shards_root,target=args.target)
    elif args.command=='validate':
        result=validate(root=args.target,verify_shards=2)
    elif args.command=='probe':
        import time,numpy as np
        from al_pipeline.astronomy.fusion_image.test_fastcache_v1 import TestFastDataset,attach_resident
        from al_pipeline.astronomy.fusion_image.contracts import load_rows
        meta=json.loads((args.target/'fast_cache_manifest.json').read_text())
        # IMPORTANT: no label column -> test evaluation is NOT performed by probe.
        rows=load_rows(meta['test_manifest'])
        ds=TestFastDataset(rows=rows,root=args.target,id_col=meta['test_id_column'],label_col=None)
        resident=attach_resident(ds,root=args.target,verify_files=True)
        samples=min(len(ds),128)
        start=time.perf_counter()
        for i in np.random.default_rng(135).choice(len(ds),size=samples,replace=False):
            x=ds[int(i)]
            if x['image'].shape!=(3,63,63) or x['metadata'].shape!=(55,):
                raise ValueError('Invalid resident loader sample')
            if 'target_index' in x:raise ValueError('Cache probe accidentally exposed labels')
        result={**resident,'samples_checked':samples,
            'random_fetch_seconds':time.perf_counter()-start,
            'label_access':False,'metrics_computed':False}
    else:
        plan=frozen_plan(args.shards_root)
        done=sum((args.shards_root/f'shard_{i:03d}'/'done.json').is_file()
                 for i in range(plan['n_shards']))
        result={'planned_shards':plan['n_shards'],'complete_shards':done,
                'consolidated':(args.target/'fast_cache_manifest.json').is_file(),
                'target':str(args.target)}
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)

if __name__=='__main__':main()
