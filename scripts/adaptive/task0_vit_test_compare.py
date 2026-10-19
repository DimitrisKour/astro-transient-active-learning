#!/usr/bin/env python3
"""Explicitly gated, provenance-audited old ViT versus new ConvNeXt test comparison.

Never imported by Task-0 trainer. No test access before all six fits complete.
Old ViT test metrics must be from a completed historical five-class evaluation,
with validated experiment manifest and checkpoint identity. We do not rescore the
old ViT checkpoint under new-model preprocessing, nor alter its train set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.fusion_image.contracts import (
    CLASS_NAMES, label_index, load_index, load_rows, object_labels, sha256, write_json_atomic,
)
from al_pipeline.astronomy.fusion_image.model import NewImageModelSpec

TASK0_ROOT = Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_task0_full_historical_v1')
BASE_SOURCE = Path('/projects/bcrv/dkourtesis/applecider_adaptive/new_image_l0_matrix_v1/plan.json')


def loaded_task0(root: Path, require_finished: bool = True) -> tuple[dict, dict]:
    """Require frozen six-fit provenance and primary loss-selected checkpoints."""
    plan_path = root/'plan.json'
    plan=json.loads(plan_path.read_text())
    assert plan['experiment'] == 'task0_full_history_btsbot_pretrained_vs_randomized'
    assert plan['test_access'] is False
    cases=plan['recipes']
    if len(cases)!=6 or {k for k in plan['run_initialization'].values()} != {'randomized','pretrained'}:
        raise ValueError('Not the complete six-run Task 0 design')
    summaries={}
    if require_finished:
        for case,recipe in cases.items():
            p=root/case/'training_summary.json'
            if not p.is_file():
                raise RuntimeError(f'All Task 0 fits must complete before test access: {case}')
            obj=json.loads(p.read_text())
            if obj.get('status')!='completed' or obj.get('identity',{}).get('plan_sha256') != sha256(plan_path):
                raise ValueError(f'Task 0 run incomplete/provenance mismatch: {case}')
            if obj.get('identity',{}).get('training_seed') != recipe['seed']:
                raise ValueError(f'Task 0 fit seed mismatch: {case}')
            if obj.get('identity',{}).get('image_initialization') != plan['run_initialization'][case]:
                raise ValueError(f'Task 0 fit initialization mismatch: {case}')
            if obj.get('checkpoint_weights') != 'raw':
                raise ValueError('Selected Task 0 checkpoint weights unexpectedly changed')
            checkpoint=root/case/'best_vit_tower.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            summaries[case]=obj
    return plan,summaries


def manifest_facts(path: Path, id_col: str, label_col: str):
    rows=load_rows(path)
    labels=object_labels(rows,id_col=id_col,label_col=label_col)
    # Canonical tuple multiset to guard identical test observations/labels even
    # if the two CSV files have different row ordering.
    if not all('alert' in r for r in rows):
        raise ValueError(f'No alert field in manifest {path}')
    stages=Counter((str(r[id_col]).strip(),int(r['alert']),label_index(r[label_col])) for r in rows)
    canonical=hashlib.sha256()
    for triple,num in sorted(stages.items()):
        canonical.update(json.dumps([*triple,num],separators=(',',':')).encode()+b'\n')
    return {'path':str(path.resolve()), 'sha256':sha256(path),
            'rows':len(rows),'objects':len(labels),
            'id_col':id_col,'label_col':label_col,
            'canonical_stage_label_multiset_sha256':canonical.hexdigest()},labels


def audit_splits(root: Path, vit_train: Path, vit_val: Path, vit_test: Path, report: Path):
    plan,_=loaded_task0(root,require_finished=True)
    if sha256(BASE_SOURCE)!=plan['source_plan_sha256']:
        raise ValueError('Original Task 0 source plan changed since frozen training')
    old=json.loads(BASE_SOURCE.read_text())
    ds=DatasetRegistry.from_toml(old['registry']).resolve_profile('astrominn_ztf').dataset
    test_path=Path(ds.test_manifest)
    if not test_path.is_file():
        raise FileNotFoundError('New model test manifest missing from audited historical registry')
    paths={'new_train':Path(ds.train_manifest), 'new_validation':Path(ds.validation_manifest),
           'new_test':test_path,
           'vit_train':vit_train,'vit_validation':vit_val,'vit_test':vit_test}
    facts,labels={},{}
    for name,path in paths.items():
        facts[name], labels[name]=manifest_facts(path,ds.object_id_column,ds.label_column)
    if facts['new_train']['sha256']!=plan['train_manifest_sha256']:
        raise ValueError('Task 0 training manifest modified since plan freezing')
    if facts['new_validation']['sha256']!=plan['validation_manifest_sha256']:
        raise ValueError('Task 0 validation manifest modified since plan freezing')
    for prefix in ('new','vit'):
        sets=[set(labels[prefix+'_'+split]) for split in ('train','validation','test')]
        if any(sets[i] & sets[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError(f'{prefix}: train/val/test overlap: test leakage risk')
    if (len(labels['new_train']),len(labels['new_validation'])) != (4729,2365):
        raise ValueError('Task 0 full historical train/validation object extent differs')
    # Strict head-to-head: identical training membership, validation membership,
    # TEST IDs and actual alert-stage labeling, not merely similar class counts.
    for split in ('train','validation','test'):
        a=labels['new_'+split];b=labels['vit_'+split]
        if a!=b:
            raise ValueError(f'ViT and new model {split} object IDs/labels differ: not a paired population')
        if facts['new_'+split]['canonical_stage_label_multiset_sha256'] != facts['vit_'+split]['canonical_stage_label_multiset_sha256']:
            raise ValueError(f'ViT and new model {split} alert-stage/label multiset differs')
    # Explicit cross-model train/test exclusions (redundant if exact match above,
    # but important as an independent safety assertion).
    test_ids=set(labels['new_test'])
    for prefix in ('new','vit'):
        if test_ids & (set(labels[prefix+'_train']) | set(labels[prefix+'_validation'])):
            raise ValueError(f'Forbidden {prefix} train/validation IDs in common test set')
    result={'schema_version':1,'purpose':'post-training_common_untouched_historical_test_audit',
            'task0_plan_sha256':sha256(root/'plan.json'),
            'class_order':list(CLASS_NAMES),
            'manifest_facts':facts,
            'matched_training_objects':len(labels['new_train']),
            'matched_validation_objects':len(labels['new_validation']),
            'matched_test_objects':len(labels['new_test']),
            'identical_manifest_stage_labels':True,
            'within_model_train_validation_test_disjoint':True,
            'vit_training_provenance_caveat':
                'These ViT manifests must be corroborated against the actual checkpoint experiment/run configuration; passing the audit alone cannot establish how an arbitrary external checkpoint was trained.',
            'test_evaluation_permitted':True}
    if report.exists():
        if json.loads(report.read_text()) != result:
            raise FileExistsError('An existing split audit differs; preserve prior audit')
        print(json.dumps(result,indent=2))
        return
    report.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(report,result)
    print(json.dumps(result,indent=2))


def verified_audit(root: Path,audit_path: Path) -> dict:
    plan,_=loaded_task0(root,require_finished=True)
    audit=json.loads(audit_path.read_text())
    if audit.get('task0_plan_sha256') != sha256(root/'plan.json') or audit.get('test_evaluation_permitted') is not True:
        raise ValueError('Test audit has invalid/frozen-plan identity')
    facts=audit['manifest_facts']
    for key in ('new_train','new_validation','new_test','vit_train','vit_validation','vit_test'):
        path=Path(facts[key]['path'])
        if sha256(path) != facts[key]['sha256']:
            raise ValueError(f'Split manifest changed after audit: {key}')
    if facts['new_test']['canonical_stage_label_multiset_sha256'] != facts['vit_test']['canonical_stage_label_multiset_sha256']:
        raise ValueError('Test populations not identical')
    return audit


def evaluate_new(root: Path, audit_path: Path, case: str, device: str, batch_size: int,
                 test_cache: Path):
    """Final-test inference on immutable, resident-RAM TEST arrays.

    This function is intentionally inaccessible until six fits complete and
    old-ViT/new-model manifest compatibility has been audited. Cache planning
    and materialization never invoke it or read labels.
    """
    audit=verified_audit(root,audit_path)
    plan,summaries=loaded_task0(root,require_finished=True)
    if case not in summaries:
        raise ValueError('Unknown Task 0 case')
    import time
    import torch
    from torch.utils.data import DataLoader
    from dataclasses import asdict
    from al_pipeline.astronomy.fusion_image.backends import metrics_for_loader
    from al_pipeline.astronomy.fusion_image.model import load_checkpoint
    from al_pipeline.astronomy.fusion_image.test_fastcache_v1 import (
        SOURCE as TEST_SOURCE, TestFastDataset, attach_resident, validate as validate_test_cache
    )
    from al_pipeline.core.evaluation.base import SplitEvaluation

    test_cache=Path(test_cache)
    # Test cache must have been built independently against EXACT frozen Task0
    # split hashes. Its stage index contains no ground-truth labels.
    test_manifest=json.loads((test_cache/'fast_cache_manifest.json').read_text())
    if test_manifest.get('source')!=TEST_SOURCE or test_manifest.get('labels_saved_in_cache') is not False:
        raise ValueError('Only the dedicated label-free TEST cache is permitted')
    if (test_manifest.get('task0_plan_sha256')!=sha256(root/'plan.json') or
        test_manifest.get('train_manifest_sha256')!=plan['train_manifest_sha256'] or
        test_manifest.get('validation_manifest_sha256')!=plan['validation_manifest_sha256'] or
        test_manifest.get('test_manifest_sha256')!=audit['manifest_facts']['new_test']['sha256']):
        raise ValueError('Cached TEST source populations do not match frozen study')
    if (test_manifest['test_rows']!=audit['manifest_facts']['new_test']['rows'] or
        test_manifest['test_objects']!=audit['manifest_facts']['new_test']['objects']):
        raise ValueError('Cached TEST population differs from audited old/new test')
    # Structural/coverage checks are cheap, mostly metadata/index I/O.
    checks=validate_test_cache(root=test_cache,verify_shards=0)

    spec=NewImageModelSpec(**{**plan['model_spec'],
            'image_initialization':plan['run_initialization'][case]})
    checkpoint=root/case/'best_vit_tower.pth'
    out=root/'final_test_evaluation'/case
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserve existing held-out evaluation: {out}')
    test_path=Path(audit['manifest_facts']['new_test']['path'])
    started=time.perf_counter()
    rows=load_rows(test_path)
    dataset=TestFastDataset(rows=rows,root=test_cache,
         id_col=test_manifest['test_id_column'],label_col=test_manifest['test_label_column'])
    if len(dataset)!=test_manifest['test_rows']:
        raise ValueError('Test manifest row count changed after caching')
    resident_info=attach_resident(dataset,root=test_cache,verify_files=False)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    model,cfg,payload=load_checkpoint(checkpoint,spec=spec,
                  seed=plan['recipes'][case]['seed'],device=torch.device(device))
    from al_pipeline.astronomy.fusion_image.cache_helpers import preprocessing_identity
    if preprocessing_identity(cfg)!=test_manifest['preprocessing_identity']:
        raise ValueError('TEST cached preprocessing differs from selected model checkpoint')
    inference_started=time.perf_counter()
    sample,obj,per_class,nrows,nobjects=metrics_for_loader(model,loader,torch.device(device))
    inference_seconds=time.perf_counter()-inference_started
    if nrows!=test_manifest['test_rows'] or nobjects!=test_manifest['test_objects']:
        raise ValueError('Incomplete final test inference coverage')
    selection=payload.get('checkpoint_metric')
    if selection!='loss':
        raise ValueError('Only preregistered loss-selected checkpoints accepted')
    result=SplitEvaluation(split='test',sample_count=nrows,object_count=nobjects,
                sample_metrics=sample,object_metrics=obj)
    # All calculations done; publish request and results atomically within
    # newly reserved per-case output directory. No existing directory overwritten.
    out.mkdir(parents=True,exist_ok=False)
    write_json_atomic(out/'test_request.json',{
        'task0_plan_sha256':sha256(root/'plan.json'),
        'split_audit_sha256':sha256(audit_path),'case':case,
        'seed':plan['recipes'][case]['seed'],
        'image_initialization':plan['run_initialization'][case],
        'checkpoint_sha256':sha256(checkpoint),
        'checkpoint_selection':'validation_loss',
        'test_manifest_sha256':sha256(test_path),
        'class_order':list(CLASS_NAMES),
        'test_cache_manifest_sha256':sha256(test_cache/'fast_cache_manifest.json'),
        'test_cache_stage_index_sha256':test_manifest['stage_index_sha256'],
        'data_access_mode':resident_info['strategy'],
    })
    write_json_atomic(out/'test_per_class_metrics.json',{
        CLASS_NAMES[int(key)]:vals for key,vals in per_class.items()})
    write_json_atomic(out/'evaluation_summary.json',{
        'checkpoint_sha256':sha256(checkpoint),
        'class_names':list(CLASS_NAMES),'splits':[asdict(result)],
        'test_cache_manifest_sha256':sha256(test_cache/'fast_cache_manifest.json'),
        'inference_seconds':inference_seconds,
        'end_to_end_seconds':time.perf_counter()-started,
        'resident_loader':resident_info,
        'cache_validation':checks,
    })
    print(json.dumps({'case':case,'test_evaluation':str(out/'evaluation_summary.json'),
              'test_objects':nobjects,'test_rows':nrows,
              'resident_load_seconds':resident_info['load_seconds'],
              'inference_seconds':inference_seconds,'object_test_metrics':obj},indent=2))


def load_vit_evaluation(path: Path, audit: dict, vit_experiment: Path, vit_checkpoint: Path):
    experiment=json.loads(vit_experiment.read_text())
    facts=audit['manifest_facts']
    if experiment.get('validation_manifest_sha256') != facts['vit_validation']['sha256']:
        raise ValueError('Old ViT experiment record does not match audited validation population')
    # In the historical experiment, the 'historical_train_manifest' is the small
    # AL L0 initial subset; it must NOT be mistaken for final full-data training.
    if experiment.get('candidate_objects') != 4253 or experiment.get('acquisition_objects') != 4253:
        raise ValueError('ViT experiment record does not correspond to complete nine-cycle AL pool')
    # Verify the ACTUAL original experiment's L0 and pool IDs reconstruct the
    # same full 4,729-object old-model TRAIN population; do not trust merely
    # a similarly named historical full-training manifest.
    id_col=facts['vit_train']['id_col']
    split_ids={}
    for key, checksum_key in (
        ('historical_train_manifest','historical_train_manifest_sha256'),
        ('candidate_manifest','candidate_manifest_sha256'),
    ):
        source=experiment.get(key)
        if not source or not Path(source).is_file():
            raise FileNotFoundError('Original ViT AL run manifest missing: '+str(source))
        if sha256(source) != experiment.get(checksum_key):
            raise ValueError(f'Original ViT AL manifest {key} SHA-256 changed')
        rows=load_rows(source)
        if id_col not in rows[0]:
            raise ValueError(f'ViT AL manifest missing expected ID column {id_col}: {source}')
        split_ids[key]={str(row[id_col]).strip() for row in rows}
        if not all(split_ids[key]):
            raise ValueError(f'Empty ViT AL object IDs in {source}')
    original_l0=split_ids['historical_train_manifest']
    candidate_pool=split_ids['candidate_manifest']
    if original_l0 & candidate_pool:
        raise ValueError('Original ViT initial L0 and candidate pool overlap')
    vit_full=manifest_facts(Path(facts['vit_train']['path']),id_col,facts['vit_train']['label_col'])[1]
    if len(original_l0)!=476 or len(candidate_pool)!=4253 or original_l0|candidate_pool != set(vit_full):
        raise ValueError('ViT cycle-9 AL train set does not reconstruct the audited full historical training population')
    vit_test_ids=set(manifest_facts(Path(facts['vit_test']['path']),id_col,facts['vit_test']['label_col'])[1])
    if (original_l0|candidate_pool) & vit_test_ids:
        raise ValueError('Original ViT AL training pool overlaps final test objects')

    obj=json.loads(path.read_text())
    if str(vit_checkpoint.resolve()) != str(Path(obj['checkpoint_path']).resolve()):
        raise ValueError('ViT evaluation checkpoint does not match explicitly supplied checkpoint path')
    if experiment.get('config',{}).get('max_cycles') != 9 or '/cycle_009/' not in str(vit_checkpoint):
        raise ValueError('Expected completed full-data cycle 9 ViT, not an early-cycle or L0 checkpoint')
    old_artifact_dir=experiment.get('artifact_dir')
    if not old_artifact_dir or not vit_checkpoint.resolve().is_relative_to(Path(old_artifact_dir).resolve()):
        raise ValueError('ViT checkpoint is not under the audited historical experiment run root')
    if not vit_checkpoint.is_file() or sha256(vit_checkpoint)!=obj.get('checkpoint_sha256'):
        raise ValueError('Old ViT checkpoint SHA-256 mismatch')
    if obj.get('model_class_names') != list(CLASS_NAMES):
        raise ValueError('Old ViT class names/order differ')
    if Path(obj.get('checkpoint_path','')).name != 'best_vit_tower.pth':
        raise ValueError('Old ViT baseline did not use the predeclared loss-selected checkpoint')
    splits={split['split']:split for split in obj['splits']}
    for split in ('validation','test'):
        if split not in splits:
            raise ValueError(f'Missing old ViT {split} evaluation')
        f=facts['vit_'+split]
        if splits[split]['object_count']!=f['objects'] or splits[split]['sample_count']!=f['rows']:
            raise ValueError(f'Old ViT evaluation count differs from audited {split} population')
    return obj,splits


def compare(root: Path,audit_path: Path,vit_eval: Path,vit_experiment: Path,vit_checkpoint: Path,output: Path):
    audit=verified_audit(root,audit_path)
    plan,summaries=loaded_task0(root,require_finished=True)
    vit,vits=load_vit_evaluation(vit_eval,audit,vit_experiment,vit_checkpoint)
    common={'macro_f1':'f1_macro','balanced_accuracy':'balanced_accuracy',
            'accuracy':'accuracy','macro_average_precision':'average_precision_macro',
            'roc_auc_macro':'roc_auc_macro','ece':'ece','log_loss':'log_loss'}
    rows=[]
    def make_row(label,model,seed,metric):
        return {'model':model,'case':label,'seed':seed,**{k:metric[v] for k,v in common.items()}}
    vit_metadata=json.loads(vit_experiment.read_text())
    rows.append(make_row('historical_vit', 'historical_vit', vit_metadata['config']['seed'], vits['test']['object_metrics']))
    for case in plan['recipes']:
        result_path=root/'final_test_evaluation'/case/'evaluation_summary.json'
        request_path=result_path.parent/'test_request.json'
        if not result_path.exists() or not request_path.exists():
            continue
        request=json.loads(request_path.read_text())
        selected=root/case/'best_vit_tower.pth'
        if request.get('task0_plan_sha256')!=sha256(root/'plan.json') or request.get('split_audit_sha256')!=sha256(audit_path):
            raise ValueError('New test evaluation from another audited experiment')
        if request.get('checkpoint_sha256')!=sha256(selected):
            raise ValueError('New test evaluated a different checkpoint')
        obj=json.loads(result_path.read_text())
        if obj.get('checkpoint_sha256')!=sha256(selected) or obj.get('class_names')!=list(CLASS_NAMES):
            raise ValueError('New evaluation summary provenance/class order mismatch')
        if obj.get('test_cache_manifest_sha256') != request.get('test_cache_manifest_sha256'):
            raise ValueError('TEST result/cache identity mismatch')
        ss=obj['splits']
        if len(ss)!=1 or ss[0]['split']!='test':
            raise ValueError('New evaluation contains unexpected splits')
        f=audit['manifest_facts']['new_test']
        if ss[0]['object_count'] != f['objects'] or ss[0]['sample_count'] != f['rows']:
            raise ValueError('New test evaluation coverage mismatch')
        rows.append(make_row(case,plan['run_initialization'][case],plan['recipes'][case]['seed'],ss[0]['object_metrics']))
    if len(rows)==1:
        raise ValueError('No completed new-model test evaluation yet; run evaluate-new first')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Comparison output already exists; preserve old results')
    output.mkdir(parents=True,exist_ok=True)
    write_json_atomic(output/'paired_test_summary.json',{
        'vit_evaluation_sha256':sha256(vit_eval),
        'vit_checkpoint_sha256':sha256(vit_checkpoint),
        'vit_experiment_sha256':sha256(vit_experiment),
        'task0_plan_sha256':sha256(root/'plan.json'),
        'audit_sha256':sha256(audit_path),
        'test_object_count':audit['matched_test_objects'],
        'comparability':'same object-stage-label test multiset; five-class labels; separate image pipelines',
        'warning':'Do not use held-out test metrics to choose recipe/checkpoint; seed/policy matching may be imperfect across architectures.',
        'rows':rows})
    with (output/'paired_test_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','evaluate-new','compare'))
    p.add_argument('--root',type=Path,default=TASK0_ROOT)
    p.add_argument('--audit-path',type=Path,default=TASK0_ROOT/'final_test_split_audit.json')
    p.add_argument('--vit-train',type=Path)
    p.add_argument('--vit-validation',type=Path)
    p.add_argument('--vit-test',type=Path)
    p.add_argument('--case',choices=[f'{ini}_s{seed}' for ini in ('pretrained','randomized') for seed in (135,246,357)])
    p.add_argument('--vit-evaluation',type=Path)
    p.add_argument('--vit-experiment',type=Path)
    p.add_argument('--vit-checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=TASK0_ROOT/'final_vit_comparison')
    p.add_argument('--test-cache',type=Path,default=Path('/work/hdd/bcrv/dkourtesis/applecider_adaptive/new_image_historical_test_fastcache_v1'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--confirm-frozen-final-test',action='store_true')
    args=p.parse_args()
    if not args.confirm_frozen_final_test:
        p.error('Explicit --confirm-frozen-final-test is required; all six fits must be complete')
    if args.command=='audit':
        if not all((args.vit_train,args.vit_validation,args.vit_test)):
            p.error('audit requires --vit-train --vit-validation --vit-test')
        audit_splits(args.root,args.vit_train,args.vit_validation,args.vit_test,args.audit_path)
    elif args.command=='evaluate-new':
        if args.case is None:p.error('evaluate-new requires --case')
        evaluate_new(args.root,args.audit_path,args.case,args.device,args.batch_size,args.test_cache)
    elif args.command=='compare':
        if not all((args.vit_evaluation,args.vit_experiment,args.vit_checkpoint)):
            p.error('compare requires --vit-evaluation --vit-experiment --vit-checkpoint')
        compare(args.root,args.audit_path,args.vit_evaluation,args.vit_experiment,args.vit_checkpoint,args.output)

if __name__=='__main__':
    main()
