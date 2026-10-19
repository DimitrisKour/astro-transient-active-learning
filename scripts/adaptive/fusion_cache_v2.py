#!/usr/bin/env python3
"""Build an incremental, restartable, shard-local L0 + validation stage cache."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.cache_v2 import prepare, build_shard, finalize
from al_pipeline.astronomy.fusion_image.contracts import class_order_config, sha256


def inputs(source_plan: Path):
    prior = json.loads(source_plan.read_text())
    upstream_sha = subprocess.check_output(['git','-C',prior['upstream_repo'],
                                            'rev-parse','HEAD'],text=True).strip()
    if upstream_sha != prior['model_sha']:
        raise ValueError('Upstream checkout modified since prior L0 plan')
    if sha256(prior['new_index']) != prior['file_hashes']['new_index']:
        raise ValueError('NPZ index differs from prior L0 plan')
    from fusion_model_basics.config import build_astrominn_config
    cfg = build_astrominn_config('btsbot_images',coord_checkpoint=None,
                 hf_image_model_dir=prior['hf_snapshot'],
                 classes=class_order_config(), device='cpu',dir='.')
    ds=DatasetRegistry.from_toml(prior['registry']).resolve_profile('astrominn_ztf').dataset
    assert ds.validation_manifest is not None
    manifests={'train':Path(prior['l0_manifest']), 'validation':Path(ds.validation_manifest)}
    if sha256(manifests['train']) != prior['file_hashes']['l0_manifest'] or \
       sha256(manifests['validation']) != prior['file_hashes']['validation_manifest']:
        raise ValueError('Historical L0 or validation manifest changed')
    return prior,ds,manifests,cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('prepare','build-shard','finalize'))
    p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--shards',type=int,default=24)
    p.add_argument('--shard-id',type=int,default=None)
    args=p.parse_args()
    prior,ds,manifests,cfg=inputs(args.source_plan)
    if args.command == 'prepare':
        result=prepare(cache_root=args.cache_root,manifests=manifests,
              index_csv=Path(prior['new_index']),runtime_config=cfg,
              id_col=ds.object_id_column,label_col=ds.label_column,
              n_shards=args.shards,upstream_sha=prior['model_sha'])
    elif args.command=='build-shard':
        if args.shard_id is None:
            p.error('build-shard needs --shard-id')
        result=build_shard(root=args.cache_root,shard_id=args.shard_id,
                           runtime_config=cfg)
    else:
        result=finalize(args.cache_root)
    print(json.dumps({k:v for k,v in result.items() if k!='assignments'},indent=2),flush=True)


if __name__=='__main__':main()
