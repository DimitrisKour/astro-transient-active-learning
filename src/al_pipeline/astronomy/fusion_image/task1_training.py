"""Task-1 cached AL fitting and automatically sealed, report-only TEST inference.

Uses the same Task-0 recipe (D4 augmentation, weighted CE, 3-epoch warmup,
cosine LR, best validation LOSS checkpoint; additional macro-F1 checkpoint).
Optimizer and scheduler begin afresh per cycle; warm-start inherits weights
only. TEST is never loaded by the fitting/checkpoint-selection code path.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from al_pipeline.astronomy.fusion_image.cache_helpers import preprocessing_identity
from al_pipeline.astronomy.fusion_image.contracts import CLASS_NAMES, load_rows, sha256, write_json_atomic
from al_pipeline.astronomy.fusion_image.fast_cache import FastHistoricalDataset
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec, build_model, load_checkpoint
from al_pipeline.astronomy.fusion_image.recipe_training_resident import (
    Recipe, augment_images, evaluate, lr_multiplier, weight_vector,
)
from al_pipeline.astronomy.fusion_image.resident_cache import attach_resident_cache


def frozen_recipe(seed: int, *, name: str = 'task1_al') -> Recipe:
    return Recipe(name=name, seed=seed, max_epochs=80, patience=15,
                  batch_size=64, learning_rate=1e-4, weight_decay=0.01,
                  class_weight_exponent=0.85, lr_schedule='warmup_cosine',
                  warmup_epochs=3, warmup_factor=0.1, cosine_min_factor=0.05,
                  image_augmentation=True, ema=False,
                  class_weighting='inverse_frequency')


def verify_training_summary(*, folder: Path, identity: dict, train_sha: str,
                            validation_sha: str) -> bool:
    f=folder/'training_summary.json'
    if not f.exists():
        return False
    data=json.loads(f.read_text())
    expected={'status':'completed','identity':identity,'train_manifest_sha256':train_sha,
              'validation_manifest_sha256':validation_sha}
    for k,v in expected.items():
        if data.get(k)!=v:
            raise ValueError(f'Completed fit provenance changed: {k}, {folder}')
    for filename in ('best_vit_tower.pth','best_object_f1.pth','training_history.json'):
        if not (folder/filename).is_file():
            raise FileNotFoundError(folder/filename)
    if data.get('primary_checkpoint_sha256') != sha256(folder/'best_vit_tower.pth'):
        raise ValueError('Completed checkpoint SHA changed')
    return True


def fit(*, spec: NewImageModelSpec, seed: int, train_manifest: Path,
        validation_manifest: Path, cache_root: Path, id_col: str, label_col: str,
        output_dir: Path, identity: dict, initial_checkpoint: Path | None,
        resident_arrays: tuple[np.ndarray,np.ndarray] | None = None):
    """Never resume or silently overwrite incomplete gradient training."""
    output_dir=Path(output_dir)
    tsha=sha256(train_manifest); vsha=sha256(validation_manifest)
    if verify_training_summary(folder=output_dir,identity=identity,
                               train_sha=tsha,validation_sha=vsha):
        return json.loads((output_dir/'training_summary.json').read_text())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Partial training is preserved for diagnosis; no overwrite: {output_dir}')
    if not torch.cuda.is_available():
        raise RuntimeError('Task-1 scientific training requires an allocated CUDA GPU')
    output_dir.mkdir(parents=True,exist_ok=False)
    recipe=frozen_recipe(seed)
    device=torch.device('cuda')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    start=time.perf_counter()
    if initial_checkpoint is None:
        model,cfg=build_model(spec,seed=seed,device=device)
    else:
        model,cfg,_=load_checkpoint(initial_checkpoint,spec=spec,seed=seed,device=device)
    root_manifest=json.loads((cache_root/'fast_cache_manifest.json').read_text())
    if preprocessing_identity(cfg)!=root_manifest['preprocessing_identity']:
        raise ValueError('Task-1 checkpoint/cached preprocessing mismatch')
    train_rows=load_rows(train_manifest);val_rows=load_rows(validation_manifest)
    tr=FastHistoricalDataset(rows=train_rows,cache_root=cache_root,split='train',
                             object_column=id_col,label_column=label_col)
    va=FastHistoricalDataset(rows=val_rows,cache_root=cache_root,split='validation',
                             object_column=id_col,label_column=label_col)
    if resident_arrays is None:
        resident=attach_resident_cache(tr,va,cache_root=cache_root)
    else:
        im,md=resident_arrays
        if im.shape!=(43309,3,63,63) or md.shape!=(43309,55) or im.dtype!=np.float32 or md.dtype!=np.float32:
            raise ValueError('Wrong shared resident training cache schema')
        for dataset in (tr,va):
            if len(dataset.positions) and max(dataset.positions)>=len(im):
                raise ValueError('Resident cache position outside bounds')
            dataset.images=im;dataset.metadata=md
        resident={'strategy':'task1_shared_acquisition_training_resident_arrays',
                  'resident_bytes':int(im.nbytes+md.nbytes),'shared_between_splits':True}
    dl=DataLoader(tr,batch_size=recipe.batch_size,shuffle=True,num_workers=0,
                  generator=torch.Generator().manual_seed(seed))
    vdl=DataLoader(va,batch_size=recipe.batch_size,shuffle=False,num_workers=0)
    w=weight_vector(train_rows,recipe=recipe,id_col=id_col,label_col=label_col)
    weights=torch.as_tensor(w,dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=recipe.learning_rate,weight_decay=recipe.weight_decay)
    # Fresh optimizer/scheduler at every cycle; the checkpoint contributes
    # ONLY model weights (never past optimizer states or hidden pool labels).
    write_json_atomic(output_dir/'fit_identity.json',identity)
    history=[];best_loss=math.inf;best_f1=-math.inf;stale=0
    best_epoch=0;best_f1_epoch=0
    train_start=time.perf_counter()
    for epoch in range(1,recipe.max_epochs+1):
        rate=recipe.learning_rate*lr_multiplier(recipe,epoch)
        for g in optimizer.param_groups:g['lr']=rate
        model.train();sum_loss=0.;batches=0;t0=time.perf_counter()
        for batch in dl:
            inp=batch['metadata'].to(device)
            image=augment_images(batch['image'].to(device))
            target=batch['target_index'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(inp,image)['logits']
                loss=F.cross_entropy(logits.float(),target,weight=weights,reduction='none').mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite AL training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();sum_loss+=float(loss.detach());batches+=1
        train_s=time.perf_counter()-t0;t1=time.perf_counter()
        val=evaluate(model,vdl,weights,device)
        val_s=time.perf_counter()-t1
        f1=float(val['object_metrics']['f1_macro']);vl=float(val['loss'])
        record={'epoch':epoch,'lr':rate,'train_loss':sum_loss/max(batches,1),
                'validation_loss':vl,'validation_macro_f1':f1,
                'training_seconds':train_s,'validation_seconds':val_s}
        history.append(record)
        write_json_atomic(output_dir/'training_history.json',history)
        print(f'[task1] epoch={epoch:02d} val_loss={vl:.6f} val_f1={f1:.5f}',flush=True)
        ck={'model_state':model.state_dict(),'model_spec':asdict(spec),
            'class_names':list(CLASS_NAMES),'seed':seed,'epoch':epoch,
            'validation_loss':vl,'validation_macro_f1':f1,
            'checkpoint_weights':'raw'}
        if f1>best_f1:
            best_f1=f1;best_f1_epoch=epoch
            p=output_dir/'best_object_f1.pth.tmp'
            torch.save({**ck,'checkpoint_metric':'object_macro_f1'},p)
            p.replace(output_dir/'best_object_f1.pth')
        if vl<best_loss:
            best_loss=vl;best_epoch=epoch;stale=0
            p=output_dir/'best_vit_tower.pth.tmp'
            torch.save({**ck,'checkpoint_metric':'loss'},p)
            p.replace(output_dir/'best_vit_tower.pth')
        else:
            stale+=1
        if stale>=recipe.patience:break
    train_seconds=time.perf_counter()-train_start
    # Evaluate PRIMARY loss-selected model, never last epoch or F1-selected.
    selected,_,_=load_checkpoint(output_dir/'best_vit_tower.pth',spec=spec,seed=seed,device=device)
    validation=evaluate(selected,vdl,weights,device)
    write_json_atomic(output_dir/'validation_per_class_metrics.json',validation['object_per_class'])
    summary={
        'status':'completed','identity':identity,
        'train_manifest_sha256':tsha,'validation_manifest_sha256':vsha,
        'checkpoint_selection':'minimum_weighted_validation_loss',
        'best_epoch':best_epoch,'best_macro_f1_epoch':best_f1_epoch,
        'best_validation_loss':best_loss,'best_macro_f1_during_fit':best_f1,
        'epochs_completed':len(history),'train_objects':len(set(r[id_col] for r in train_rows)),
        'train_rows':len(train_rows),'validation_rows':len(val_rows),
        'validation_object_metrics':validation['object_metrics'],
        'validation_sample_metrics':validation['sample_metrics'],
        'validation_per_class':validation['object_per_class'],
        'primary_checkpoint_sha256':sha256(output_dir/'best_vit_tower.pth'),
        'secondary_checkpoint_sha256':sha256(output_dir/'best_object_f1.pth'),
        'initial_checkpoint_sha256':sha256(initial_checkpoint) if initial_checkpoint else None,
        'class_weights':dict(zip(CLASS_NAMES,map(float,w))),
        'fit_seconds':train_seconds,'wall_seconds':time.perf_counter()-start,
        'resident_cache':resident,
    }
    write_json_atomic(output_dir/'training_summary.json',summary)
    return summary


def infer_acquisition(*,checkpoint: Path,spec: NewImageModelSpec,seed: int,
                      stage_rows: list[dict],cache_root: Path,batch_size:int=64,
                      resident_arrays: tuple[np.ndarray,np.ndarray] | None = None):
    """Pool inference uses only object/stage keys, never ground truth."""
    if not stage_rows:
        raise ValueError('Empty unlabeled pool')
    if any(set(row)-{'object_id','alert'} for row in stage_rows):
        raise ValueError('Candidate inference must never receive label columns')
    device=torch.device('cuda')
    model,cfg,_=load_checkpoint(checkpoint,spec=spec,seed=seed,device=device)
    if preprocessing_identity(cfg)!=json.loads((cache_root/'fast_cache_manifest.json').read_text())['preprocessing_identity']:
        raise ValueError('Pool preprocessing identity mismatch')
    ds=FastHistoricalDataset(rows=stage_rows,cache_root=cache_root,split='train',
          object_column='object_id',label_column=None)
    # Single sequential host memory load, NO repeated NPZ reads.
    if resident_arrays is None:
        image=np.load(cache_root/'images.npy',allow_pickle=False)
        md=np.load(cache_root/'metadata.npy',allow_pickle=False)
    else:
        image,md=resident_arrays
    if isinstance(image,np.memmap) or isinstance(md,np.memmap):
        raise ValueError('Pool data are not resident')
    image.setflags(write=False);md.setflags(write=False)
    ds.images=image;ds.metadata=md
    dl=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    ids=[];logits=[];features=[]
    model.eval()
    with torch.inference_mode():
        for batch in dl:
            out=model(batch['metadata'].to(device),batch['image'].to(device),return_features=True)
            if not hasattr(out,'logits') or not hasattr(out,'embedding'):
                raise ValueError('Model does not expose logits and embeddings for AL acquisition')
            scores=out.logits; emb=out.embedding
            if scores.ndim!=2 or scores.shape[1]!=5 or emb.ndim!=2 or emb.shape[0]!=len(scores):
                raise ValueError('Bad AL model prediction shape')
            if not torch.isfinite(scores).all() or not torch.isfinite(emb).all():
                raise ValueError('Nonfinite pool logits or embedding')
            ids.extend(batch['obj_id']);logits.append(scores.float().cpu().numpy())
            features.append(emb.float().cpu().numpy())
    if ids!=[r['object_id'] for r in stage_rows]:
        raise ValueError('Predicted object ordering differs from immutable candidate ordering')
    return ids,np.concatenate(logits),np.concatenate(features)


def sealed_test(*,checkpoint:Path,spec:NewImageModelSpec,seed:int,
                test_manifest:Path,test_cache:Path,id_col:str,label_col:str,
                output_dir:Path,expected_test_sha:str,batch_size:int=64):
    """TEST ONLY after training complete; emits one sealed reporting artifact.

    This function is never called by acquisition or by checkpoint selection.
    The next cycle reads only complete.json, not this result file.
    """
    if sha256(test_manifest)!=expected_test_sha:
        raise ValueError('Unexpected test manifest')
    cache=json.loads((test_cache/'fast_cache_manifest.json').read_text())
    if cache['test_manifest_sha256']!=expected_test_sha or cache.get('labels_saved_in_cache') is not False:
        raise ValueError('Incompatible or label-bearing test cache')
    from al_pipeline.astronomy.fusion_image.test_fastcache_v1 import (
        TestFastDataset,attach_resident,
    )
    from al_pipeline.astronomy.fusion_image.backends import metrics_for_loader
    if output_dir.exists() and (output_dir/'evaluation_summary.json').is_file():
        old=json.loads((output_dir/'evaluation_summary.json').read_text())
        if old['checkpoint_sha256']!=sha256(checkpoint) or old['test_manifest_sha256']!=expected_test_sha:
            raise ValueError('Test evaluation identity differs; refusing reuse')
        return old
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Partial test results require inspection: {output_dir}')
    device=torch.device('cuda');torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    model,_,_=load_checkpoint(checkpoint,spec=spec,seed=seed,device=device)
    rows=load_rows(test_manifest)
    ds=TestFastDataset(rows=rows,root=test_cache,id_col=id_col,label_col=label_col)
    resident=attach_resident(ds,root=test_cache,verify_files=True)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)
    sample,obj,per_class,ns,no=metrics_for_loader(model,loader,device)
    if no!=2365:
        raise ValueError('Wrong number of test objects')
    result={'status':'completed','split':'test',
            'checkpoint_sha256':sha256(checkpoint),
            'test_manifest_sha256':expected_test_sha,
            'object_count':no,'sample_count':ns,
            'object_metrics':obj,'sample_metrics':sample,
            'per_class_metrics':{CLASS_NAMES[int(i)]:v for i,v in per_class.items()},
            'resident_cache':resident,'report_only':True,
            'no_selection_or_training_uses_test':True}
    output_dir.mkdir(parents=True,exist_ok=True)
    write_json_atomic(output_dir/'evaluation_summary.json',result)
    return result
