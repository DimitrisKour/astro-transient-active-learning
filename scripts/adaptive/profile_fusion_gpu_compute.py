#!/usr/bin/env python3
"""Isolate model forward/backward cost from NPZ and DataLoader overhead on one A40.

Synthetic inputs are for timing ONLY, never performance or scientific evaluation.
"""
from __future__ import annotations
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model
from al_pipeline.astronomy.fusion_image.contracts import write_json_atomic

PLAN=Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_matrix_v1/plan.json')
OUT=Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_gpu_compute_probe_v1')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU benchmark requires allocated CUDA GPU')
    if OUT.exists() and (OUT/'result.json').exists():
        raise FileExistsError('GPU compute result already exists')
    OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads(PLAN.read_text())
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    spec=NewImageModelSpec(variant='btsbot_images',image_initialization='pretrained',
          hf_snapshot=prior['hf_snapshot'],hf_weights_sha256=prior['hf_weights_sha256'],
          source_revision=prior['model_sha'])
    model,_=build_model(spec,seed=135,device='cuda')
    weights=torch.ones(5,device='cuda')
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                     lr=1e-4,weight_decay=0.01)
    batch=64
    x=torch.randn(batch,55,device='cuda',dtype=torch.float32)
    image=torch.randn(batch,3,63,63,device='cuda',dtype=torch.float32)
    labels=torch.randint(0,5,(batch,),device='cuda')

    def train_once():
        model.train()
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,image)['logits']
            loss=torch.nn.functional.cross_entropy(logits,labels,weight=weights,reduction='none').mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step()

    def val_once():
        model.eval()
        with torch.inference_mode():
            logits=model(x,image)['logits']
            return logits.shape

    for _ in range(3):train_once()
    torch.cuda.synchronize()
    started=time.perf_counter()
    n_train=10
    for _ in range(n_train):train_once()
    torch.cuda.synchronize()
    training_elapsed=time.perf_counter()-started

    for _ in range(3):val_once()
    torch.cuda.synchronize()
    started=time.perf_counter()
    n_val=20
    for _ in range(n_val):val_once()
    torch.cuda.synchronize()
    validation_elapsed=time.perf_counter()-started
    output={
        'mode':'synthetic_GPU_compute_only',
        'model_source_sha':prior['model_sha'],
        'gpu':torch.cuda.get_device_name(0),
        'batch_size':batch,
        'train_batch_seconds':training_elapsed/n_train,
        'validation_batch_seconds':validation_elapsed/n_val,
        'estimated_full_L0_training_compute_seconds':math.ceil(2890/batch)*training_elapsed/n_train,
        'estimated_full_validation_compute_seconds':math.ceil(14148/batch)*validation_elapsed/n_val,
        'estimated_compute_only_epoch_seconds':(
             math.ceil(2890/batch)*training_elapsed/n_train+
             math.ceil(14148/batch)*validation_elapsed/n_val),
        'important':'Synthetic performance lower bound; excludes NPZ reading, DataLoader, metrics, checkpoint and startup.',
    }
    write_json_atomic(OUT/'result.json',output)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
