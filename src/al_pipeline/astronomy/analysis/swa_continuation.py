"""Task9B SWA: ten-epoch continuation of an ORIGINAL rejected Task4 ViT tower.

Uses Task4's frozen data pipeline, augmentation, weighted loss and AdamW settings;
changes only learning rate to 2e-5 and holds it constant for ten epochs.
Snapshots epochs 6..10 into a single averaged ViT. No validation tuning.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np

EPOCHS=10
START_AVERAGING=6
CONTINUATION_LR=2e-5


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_torch(path,payload):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    torch.save(payload,temp);temp.replace(path)


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def static_contract(inp:dict, case:dict)->dict:
    return {'case':case,'original_candidate_checkpoint':inp['candidate_checkpoint'],
        'original_checkpoint_sha256':inp['candidate_checkpoint_sha256'],
        'train_manifest':inp['training_manifest'],'train_manifest_sha256':inp['manifest_sha256'],
        'validation_manifest':inp['validation_manifest'],'validation_manifest_sha256':inp['validation_sha256'],
        'l0_manifest':inp['l0_manifest'],'l0_manifest_sha256':hash_file(inp['l0_manifest']),
        'source_training_summary_sha256':inp['source_training_summary_sha256'],
        'source_policy':inp['model_policy'],'epochs':EPOCHS,'averaging_epochs':list(range(START_AVERAGING,EPOCHS+1)),
        'constant_learning_rate':CONTINUATION_LR,'optimizer':'AdamW freshly initialized',
        'scheduler':'none','early_stopping':False,'seed':case['seed'],
        'test_evaluation':False}


def checkpoint_state(model):
    """Copy all ViT parameters and buffers, retaining exact original key names."""
    return {key:value.detach().cpu().clone() for key,value in model.vit_tower.state_dict().items()}


def assert_swa_architecture(vit):
    import torch
    if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in vit.modules()):
        raise ValueError('ViT contains BatchNorm; SWA would require a separate BN refresh pass')


def run_ten_epochs(*, inp:dict, case:dict, output:Path, registry:Path):
    """Idempotent per-epoch resume. Model checkpoint provenance is verified on restart."""
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.swa_utils import AveragedModel
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.astrominn import training as T
    from al_pipeline.core.training.base import TrainingRequest

    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    contract=static_contract(inp,case)
    contract_path=output/'contract.json';state_path=output/'continuation_state.pth'
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding='utf-8'))!=contract:
            raise ValueError('SWA scientific contract changed in existing directory')
    else:
        if any(output.iterdir()):raise RuntimeError('Nonempty SWA directory lacks contract; inspect it')
        atomic_json(contract_path,contract)

    for p,digest in ((inp['candidate_checkpoint'],inp['candidate_checkpoint_sha256']),
                     (inp['training_manifest'],inp['manifest_sha256']),
                     (inp['validation_manifest'],inp['validation_sha256']),
                     (inp['source_train_summary'],inp['source_training_summary_sha256'])):
        if hash_file(p)!=digest:raise ValueError(f'SWA source artifact has changed: {p}')
    if not torch.cuda.is_available():raise RuntimeError('Task9B must run with CUDA')
    original=DatasetRegistry.from_toml(registry).resolve_profile('astrominn_ztf').dataset
    dataset=replace(original,train_manifest=Path(inp['l0_manifest']))
    source_policy=T.AstroMiNNViTTrainingConfig(**inp['model_policy'])
    if (source_policy.class_weight_exponent!=.85 or source_policy.class_weight_source!='training_manifest'
        or source_policy.new_object_weighting!='per_sample' or source_policy.augment_validation
        or source_policy.legacy_seed_sequence):
        raise ValueError('Task9B source training configuration mismatch')
    policy=replace(source_policy,learning_rate=CONTINUATION_LR,train_num_workers=0,validation_num_workers=0)
    trainer=T.AstroMiNNViTTrainer(dataset=dataset,policy=policy,device='cuda',initialization='pretrained',
                                   artifact_dir=Path(inp['model_artifact_dir']))
    seed=int(case['seed']);trainer._seed_process(seed)
    weights,_=T._new_object_training_weights(train_manifest=Path(inp['training_manifest']),
        historical_manifest=dataset.train_manifest,object_id_column=dataset.object_id_column,mode=policy.new_object_weighting)
    train_data=T.AstroMiNNViTManifestDataset(manifest_path=inp['training_manifest'],dataset=dataset,
                 artifact_dir=Path(inp['model_artifact_dir']),augment=policy.augment_training,
                 cache_in_memory=policy.cache_samples_in_memory,object_sample_weights=weights)
    generator=torch.Generator();generator.manual_seed(policy.loader_seed)
    loader=DataLoader(train_data,batch_size=policy.batch_size,shuffle=True,num_workers=0,
                      pin_memory=True,generator=generator)
    request=TrainingRequest(train_manifest=Path(inp['training_manifest']),validation_manifest=Path(inp['validation_manifest']),
        output_dir=output,run_id='task9b_swa_'+case['policy']+'_s'+str(seed),
        cycle=int(case['rejected_cycle']),strategy=case['policy'],seed=seed,
        initial_checkpoint=Path(inp['candidate_checkpoint']))
    model=trainer._build_model(request).to('cuda')
    T._freeze_expert_towers(model)
    assert_swa_architecture(model.vit_tower)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=AdamW(parameters,lr=CONTINUATION_LR,weight_decay=policy.weight_decay,
                    betas=(policy.beta1,policy.beta2),eps=policy.optimizer_epsilon)
    weights_class,_=T._class_weights(policy=policy,device=torch.device('cuda'),
                         train_manifest=Path(inp['training_manifest']),label_column=dataset.label_column)
    criterion=torch.nn.CrossEntropyLoss(weight=weights_class,reduction='none')
    averaged=AveragedModel(model.vit_tower,use_buffers=True,device='cuda')
    start=1;history=[]
    if state_path.exists():
        # Locally produced trusted checkpoint only. Never load untrusted PTH files.
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        if state['contract_sha256']!=hash_file(contract_path):raise ValueError('SWA resume contract changed')
        start=int(state['epoch'])+1
        if not 1<=start<=EPOCHS+1:raise ValueError('Invalid SWA resume epoch')
        model.vit_tower.load_state_dict(state['vit'])
        optimizer.load_state_dict(state['optimizer'])
        averaged.module.load_state_dict(state['swa_vit'])
        averaged.n_averaged.fill_(int(state['n_averaged']))
        generator.set_state(state['generator'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        history=state['history']
    started=time.perf_counter()
    for epoch in range(start,EPOCHS+1):
        T._set_training_mode(model,legacy_frozen_expert_train_mode=policy.legacy_frozen_expert_train_mode)
        loss=T._train_epoch(loader=loader,model=model,criterion=criterion,optimizer=optimizer,
                            trainable_parameters=parameters,device=torch.device('cuda'),gradient_clip_norm=policy.gradient_clip_norm)
        if epoch>=START_AVERAGING:averaged.update_parameters(model.vit_tower)
        history.append({'epoch':epoch,'train_loss':loss,'swa_members':int(averaged.n_averaged.item())})
        atomic_torch(state_path,{'epoch':epoch,'contract_sha256':hash_file(contract_path),
            'vit':checkpoint_state(model),'swa_vit':{k:v.detach().cpu().clone() for k,v in averaged.module.state_dict().items()},
            'n_averaged':int(averaged.n_averaged.item()),'optimizer':optimizer.state_dict(),
            'generator':generator.get_state(),'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'history':history})
    if int(averaged.n_averaged.item())!=EPOCHS-START_AVERAGING+1:
        raise ValueError('Unexpected number of SWA members')
    final=output/'final_vit_tower.pth';swa=output/'swa_vit_tower.pth'
    if not final.exists():atomic_torch(final,checkpoint_state(model))
    if not swa.exists():atomic_torch(swa,{k:v.detach().cpu().clone() for k,v in averaged.module.state_dict().items()})
    # Confirm checkpoint and averaged parameter/buffer key shapes are identical.
    final_state=torch.load(final,map_location='cpu',weights_only=True)
    swa_state=torch.load(swa,map_location='cpu',weights_only=True)
    if (final_state.keys()!=swa_state.keys() or
        any(final_state[k].shape!=swa_state[k].shape for k in final_state)):
        raise ValueError('Final/SWA checkpoint structure differs')
    return {'history':history,'training_seconds_this_launch':time.perf_counter()-started,
            'final_checkpoint':str(final),'final_sha256':hash_file(final),
            'swa_checkpoint':str(swa),'swa_sha256':hash_file(swa),
            'swa_members':int(averaged.n_averaged.item()), 'contract_sha256':hash_file(contract_path),
            'resumed':start>1}
