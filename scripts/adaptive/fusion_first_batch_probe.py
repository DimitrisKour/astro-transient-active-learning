#!/usr/bin/env python3
"""Short, non-scientific GPU diagnostic for the full historical fast-cache training path.

No checkpoints, oracle calls, test access, or changes to frozen pre-Task-0 fits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.contracts import CLASS_NAMES, label_index, load_rows, write_json_atomic
from al_pipeline.astronomy.fusion_image.fast_cache import FastHistoricalDataset
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model

BASE = Path('/projects/bcrv/dkourtesis/applecider_adaptive')
PLAN = BASE / 'new_image_pre_task0_recipe_fastcache_v2/plan.json'
OUT = BASE / 'new_image_first_batch_probe_v1'


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


def now():
    return time.perf_counter()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, default=PLAN)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--train-batches', type=int, default=8)
    p.add_argument('--val-batches', type=int, default=4)
    args = p.parse_args()
    if args.train_batches < 1 or args.val_batches < 1:
        raise ValueError('Need positive batch limits')
    if not torch.cuda.is_available():
        raise RuntimeError('Need one CUDA GPU')
    if (args.out / 'probe.json').exists():
        raise FileExistsError('Previous completed probe exists; use a new --out')
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    assert plan['training_rows'] == 29161 and plan['validation_rows'] == 14148
    assert plan['test_access'] is False and plan['acquisition'] is False
    old = json.loads(Path(plan['source_plan']).read_text())
    ds = DatasetRegistry.from_toml(old['registry']).resolve_profile('astrominn_ztf').dataset
    id_col, label_col = ds.object_id_column, ds.label_column
    train_rows = load_rows(Path(plan['train_manifest']))
    val_rows = load_rows(Path(plan['validation_manifest']))
    cache = Path(plan['cache_root'])
    print(f'[probe] rows train={len(train_rows)} val={len(val_rows)} cache={cache}', flush=True)
    assert (len(train_rows), len(val_rows)) == (29161,14148)
    train_ds = FastHistoricalDataset(rows=train_rows, cache_root=cache, split='train',
                                     object_column=id_col, label_column=label_col)
    val_ds = FastHistoricalDataset(rows=val_rows, cache_root=cache, split='validation',
                                   object_column=id_col, label_column=label_col)
    batch_size = 64
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(135), num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    dev = torch.device('cuda')
    spec = NewImageModelSpec(**plan['model_spec'])
    init_started = now()
    model, cfg = build_model(spec, seed=135, device=dev)
    init_time = now()-init_started
    print(f'[probe] model_init_seconds={init_time:.3f}; testing {args.train_batches} train + {args.val_batches} val batches', flush=True)
    counts = Counter(label_index(r[label_col]) for r in train_rows)
    cls = np.array([counts.get(i, 0) for i in range(len(CLASS_NAMES))], dtype=np.float64)
    w = cls ** (-0.85)
    w /= w.mean()
    weights = torch.as_tensor(w,dtype=torch.float32,device=dev)
    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=1e-4,weight_decay=.01)
    results = {'train': [], 'validation': []}
    it = iter(train_dl)
    model.train()
    for i in range(args.train_batches):
        print(f'[probe][train][{i+1}/{args.train_batches}] fetch START',flush=True)
        t = now()
        batch = next(it)
        fetched = now()-t
        print(f'[probe][train][{i+1}] fetch END {fetched:.3f}s; transfer START',flush=True)
        t=clock()
        x=batch['metadata'].to(dev)
        img=batch['image'].to(dev)
        target=batch['target_index'].to(dev)
        transfer=clock()-t
        print(f'[probe][train][{i+1}] transfer END {transfer:.3f}s; forward START',flush=True)
        opt.zero_grad(set_to_none=True)
        t=clock()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,img)['logits']
            loss=F.cross_entropy(logits.float(),target,weight=weights,reduction='none').mean()
        forward=clock()-t
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at diagnostic batch {i+1}')
        print(f'[probe][train][{i+1}] forward END {forward:.3f}s; backward START',flush=True)
        t=clock()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        backward=clock()-t
        r={'batch':i+1,'fetch_seconds':fetched,'transfer_seconds':transfer,'forward_seconds':forward,
           'backward_optimizer_seconds':backward,'loss':float(loss.detach())}
        results['train'].append(r)
        print(f'[probe][train][{i+1}] {r}',flush=True)
        write_json_atomic(args.out/'partial.json',results)
    model.eval()
    it=iter(val_dl)
    with torch.inference_mode():
        for i in range(args.val_batches):
            print(f'[probe][val][{i+1}/{args.val_batches}] fetch START',flush=True)
            t=now()
            batch=next(it)
            fetched=now()-t
            print(f'[probe][val][{i+1}] fetch END {fetched:.3f}s; forward START',flush=True)
            t=clock()
            logits=model(batch['metadata'].to(dev),batch['image'].to(dev))['logits']
            forward=clock()-t
            r={'batch':i+1,'fetch_seconds':fetched,'transfer_and_forward_seconds':forward}
            results['validation'].append(r)
            print(f'[probe][val][{i+1}] {r}',flush=True)
            write_json_atomic(args.out/'partial.json',results)
    total_batches=math.ceil(len(train_rows)/batch_size)
    estimate=sum(sum(r[k] for k in ('fetch_seconds','transfer_seconds','forward_seconds','backward_optimizer_seconds')) for r in results['train'])/len(results['train'])*total_batches
    summary={'train_rows':len(train_rows),'validation_rows':len(val_rows),'train_batches_per_epoch':total_batches,
             'first_batches_extrapolated_train_epoch_seconds':estimate,
             'warning':'ROUGH diagnostic extrapolation; includes GPU first-use warmup and cache cold/hot effects; not an actual epoch timing',
             'model_init_seconds':init_time,'batches':results}
    write_json_atomic(args.out/'probe.json',summary)
    print('[probe] COMPLETE '+json.dumps({k:v for k,v in summary.items() if k!='batches'},indent=2),flush=True)

if __name__=='__main__':
    main()
