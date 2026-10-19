#!/usr/bin/env python3
"""One *full* historical L0 epoch + full validation with cached inputs; no test."""
from __future__ import annotations
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.backends import ImageTrainingPolicy, NewImageTrainer
from al_pipeline.astronomy.fusion_image.contracts import load_index, sha256
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model
from al_pipeline.core.training.base import TrainingRequest

BASE=Path('/projects/bcrv/dkourtesis/applecider_adaptive')
CACHE=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_l0_cache_v3')
OLD=BASE/'new_image_l0_matrix_v1'
OUTPUT=BASE/'new_image_l0_cached_benchmark_v3'


def main():
    if not torch.cuda.is_available():raise RuntimeError('Benchmark requires CUDA GPU')
    os.environ['FUSION_IMAGE_CACHE_ROOT']=str(CACHE)
    os.environ['FUSION_IMAGE_CACHE_PRELOAD']='1'
    plan=json.loads((OLD/'plan.json').read_text())
    ds=DatasetRegistry.from_toml(plan['registry']).resolve_profile('astrominn_ztf').dataset
    model_sha=subprocess.check_output(['git','-C',plan['upstream_repo'],'rev-parse','HEAD'],text=True).strip()
    if model_sha != plan['model_sha']:raise ValueError('Model source SHA differs')
    cache=json.loads((CACHE/'cache_manifest.json').read_text())
    if cache['new_index_sha256']!=sha256(plan['new_index']):
        raise ValueError('Cache uses a different object index')
    for split, src in (('train', Path(plan['l0_manifest'])),
                       ('validation', ds.validation_manifest)):
        if cache['manifest_sha256'][split]!=sha256(src):
            raise ValueError('Cached dataset does not match historical L0/validation')
    spec=NewImageModelSpec(variant='btsbot_images',image_initialization='pretrained',
          hf_snapshot=plan['hf_snapshot'],hf_weights_sha256=plan['hf_weights_sha256'],
          source_revision=plan['model_sha'])
    policy=ImageTrainingPolicy(max_epochs=1,patience=1,batch_size=64,
          learning_rate=1e-4,weight_decay=0.01,class_weight_exponent=0.85,
          weighting='per_sample',checkpoint_metric='loss')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK',4)))
    import time
    wall_started=time.perf_counter()
    model,cfg=build_model(spec,seed=135,device='cuda')
    del model
    trainer=NewImageTrainer(spec=spec,policy=policy,index=load_index(plan['new_index']),
            runtime_config=cfg,id_col=ds.object_id_column,label_col=ds.label_column,
            device='cuda',initialization='random')
    request=TrainingRequest(train_manifest=Path(plan['l0_manifest']),
          validation_manifest=ds.validation_manifest,output_dir=OUTPUT,
          run_id='fusion_l0_cached_benchmark_pretrained_seed135',
          cycle=0,strategy='benchmark',seed=135,initial_checkpoint=None)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f'Refusing overwrite of benchmark: {OUTPUT}')
    result=trainer.train(request)
    history=json.loads(result.history_path.read_text())
    print(json.dumps({'result':'cached_full_L0_epoch_completed',
          'end_to_end_wall_seconds':time.perf_counter()-wall_started,
          'training_seconds':history[-1]['training_seconds'],
          'validation_seconds':history[-1]['validation_seconds'],
          'epoch_seconds':history[-1]['epoch_seconds'],
          'train_loss':history[-1]['train_loss'],
          'validation_loss':history[-1]['validation_loss'],
          'validation_object_macro_f1':history[-1]['validation_object_macro_f1'],
          'checkpoint':str(result.checkpoint_path)},indent=2),flush=True)

if __name__=='__main__':main()
