"""Pure Task-1 contracts; no historical labels, CUDA, or real cluster needed."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from al_pipeline.astronomy.fusion_image.task1_acquisition import (
    PHASE1,PHASE2,POLICIES,acquisition_seed,choose,
)

SCRIPTS=Path(__file__).resolve().parents[3]/'scripts'/'adaptive'
sys.path.insert(0,str(SCRIPTS))
import task1_closed_loop as al  # noqa: E402
import task1_schedule as schedule  # noqa: E402


def _example(n=80):
    rng=np.random.default_rng(101)
    ids=[f'candidate_{i:04d}' for i in range(n)]
    logits=rng.normal(size=(n,5))
    embedding=rng.normal(size=(n,12))
    return ids,logits,embedding


@pytest.mark.parametrize('policy',POLICIES)
def test_selection_is_seeded_unique_and_label_free(policy):
    ids,logits,embedding=_example()
    params={'policy':policy,'ids':ids,'logits':logits,'embeddings':embedding,
            'k':15,'seed':135,'cycle':3}
    a=choose(**params);b=choose(**params)
    assert a==b
    assert a['selected_ids']==[ids[i] for i in a['selected_indices']]
    assert len(set(a['selected_ids']))==15
    assert all(isinstance(y,int) and 0<=y<5 for y in a['selected_predicted_class'])
    assert sum(a['selected_class_composition'].values())==15
    assert sum(a['pool_class_composition'].values())==80
    assert not any('label' in key or 'ground_truth' in key for key in a)
    assert a['acquisition_rng_seed']==acquisition_seed(seed=135,cycle=3)


def test_policy_phase_partition():
    assert set(PHASE1)=={'random','entropy'}
    assert set(PHASE2)=={'entropy_diversity','entropy_balanced_explore'}
    assert set(PHASE1).isdisjoint(PHASE2)
    assert set(POLICIES)==set(PHASE1)|set(PHASE2)
    assert al.CONFIGS['random_l0']['continuation']=='restart_from_l0'
    assert al.CONFIGS['random_warm']['continuation']=='warm_start'
    assert al.CONFIGS['pretrained_warm']['initialization']=='pretrained'
    assert al.CYCLES==9 and al.ACQUISITION_SIZE==473
    assert 476+8*473+469==4729


def test_history_oracle_manifest_preserves_full_rows():
    rows=[{'obj_id':'a','alert':'0','target':'AGN'},
          {'obj_id':'a','alert':'1','target':'AGN'},
          {'obj_id':'b','alert':'0','target':'CV'},
          {'obj_id':'c','alert':'0','target':'SN Ia'}]
    u=al.stage_rows_no_labels(rows,'obj_id',{'a'})
    assert u==[{'object_id':'b','alert':'0'},{'object_id':'c','alert':'0'}]
    assert all(set(r)=={'object_id','alert'} for r in u)
    assert al.all_train_rows_for_ids(rows,{'a','c'},'obj_id')==[rows[0],rows[1],rows[3]]
    with pytest.raises(ValueError):
        al.all_train_rows_for_ids(rows,{'absent'},'obj_id')


def test_frozen_manifest_refuses_silent_row_mutation(tmp_path):
    p=tmp_path/'manifest.csv'
    rows=[{'object_id':'a','alert':'1'}]
    al.write_csv(p,rows,['object_id','alert'])
    al.write_csv(p,rows,['object_id','alert'])
    with pytest.raises(ValueError):
        al.write_csv(p,[{'object_id':'a','alert':'2'}],['object_id','alert'])


def test_scheduler_registry_and_dependency_without_sbatch(tmp_path,monkeypatch):
    root=tmp_path/'task1'
    root.mkdir()
    repo=tmp_path/'repo';script=repo/'scripts/adaptive/task1_cycle.sbatch'
    script.parent.mkdir(parents=True);script.write_text('#!/bin/bash\n')
    plan={'repo':str(repo),'test_cache':str(tmp_path/'cache')}
    monkeypatch.setattr(schedule,'verify_plan',lambda *args,**kwargs:plan)
    from types import ModuleType
    testcache_stub=ModuleType('al_pipeline.astronomy.fusion_image.test_fastcache_v1')
    testcache_stub.validate=lambda **kwargs:{'validated':True}
    monkeypatch.setitem(sys.modules,'al_pipeline.astronomy.fusion_image.test_fastcache_v1',testcache_stub)
    calls=[]
    class Completed:
        def __init__(self,number):self.stdout=f'{number}\n';self.stderr=''
    def fake_sbatch(cmd,**kw):
        assert cmd[0]=='sbatch'
        calls.append(cmd)
        return Completed(2000+len(calls))
    monkeypatch.setattr(schedule.subprocess,'run',fake_sbatch)
    schedule.submit(root=root,mode='phase1',limit=7)
    first=schedule.load_jobs(root/'slurm_jobs.tsv')
    assert len(first)==7
    assert sum(k[0]=='l0' for k in first)==6
    assert len(calls)==7
    schedule.submit(root=root,mode='phase1',limit=6)
    nextjobs=schedule.load_jobs(root/'slurm_jobs.tsv')
    assert len(nextjobs)==13 and len(calls)==13
    assert any('--dependency=afterok:2001' in c for c in calls)
    with pytest.raises(RuntimeError,match='Cannot release advanced policies'):
        schedule.submit(root=root,mode='phase2',limit=1)


def test_test_predictions_separated_from_default_reporting(tmp_path,monkeypatch):
    root=tmp_path/'task1';root.mkdir()
    monkeypatch.setattr(al,'verify_plan',lambda *a,**k:{'repo':'fake'})
    al.report(root,include_test=False)
    assert (root/'task1_validation_report.csv').is_file()
    text=(root/'task1_validation_report.csv').read_text()
    assert 'test_macro_f1' not in text
    with pytest.raises(ValueError,match='embargoed'):
        al.report(root,include_test=True)


def test_cycle_zero_and_two_cycles_chain_without_cuda(tmp_path,monkeypatch):
    """Synthetic end-to-end orchestration (fake model/fit), no oracle leakage."""
    import hashlib
    from types import ModuleType
    from al_pipeline.astronomy.fusion_image.contracts import sha256,write_json_atomic

    root=tmp_path/'task1';root.mkdir()
    train=tmp_path/'historical_train.csv'
    full=[{'obj_id':f'obj{i:05d}','alert':'0',
           'target':('AGN','TDE','SN Ia','SN II','CV')[i%5]} for i in range(4729)]
    al.write_csv(train,full,list(full[0]))
    initial={r['obj_id'] for r in full[:476]}
    pool=al.stage_rows_no_labels(full,'obj_id',initial)
    al.write_csv(root/'pool_stages.csv',pool,['object_id','alert'])
    (root/'plan.json').write_text('{"dummy":"plan"}')
    dummy=tmp_path/'test.csv';dummy.write_text('obj_id,alert,target\nobj50000,0,AGN\n')
    valid=tmp_path/'val.csv';valid.write_text('obj_id,alert,target\nobj60000,0,AGN\n')
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'fast_cache_manifest.json').write_text('{}')
    plan={'l0_ids':sorted(initial),'id_col':'obj_id','label_col':'target',
          'full_train_manifest':str(train),'validation_manifest':str(valid),
          'test_manifest':str(dummy),'test_sha256':sha256(dummy),
          'test_cache':str(tmp_path/'testcache'),'train_cache':str(cache),
          'model_specs':{
              'randomized':{'variant':'btsbot_images','image_initialization':'randomized',
                  'hf_snapshot':'/fake','source_revision':'abc'},
              'pretrained':{'variant':'btsbot_images','image_initialization':'pretrained',
                  'hf_snapshot':'/fake','source_revision':'abc'}}}
    monkeypatch.setattr(al,'verify_plan',lambda *a,**k:plan)
    monkeypatch.setattr(al,'resident_arrays',lambda root:(None,None))

    fake=ModuleType('al_pipeline.astronomy.fusion_image.task1_training')
    def fake_fit(**kwargs):
        target=kwargs['output_dir'];target.mkdir(parents=True,exist_ok=True)
        ck=target/'best_vit_tower.pth';ck.write_bytes(b'pretend-checkpoint')
        summary={'primary_checkpoint_sha256':sha256(ck),
                 'best_epoch':3,'fit_seconds':1.0,
                 'validation_object_metrics':{'f1_macro':0.5,'balanced_accuracy':0.6,'accuracy':0.7}}
        write_json_atomic(target/'training_summary.json',summary)
        return summary
    def fake_predict(**kwargs):
        rows=kwargs['stage_rows']
        return [r['object_id'] for r in rows],np.zeros((len(rows),5)),np.ones((len(rows),4))
    def fake_test(**kwargs):
        output=kwargs['output_dir'];output.mkdir(parents=True,exist_ok=True)
        write_json_atomic(output/'evaluation_summary.json',{'object_metrics':{'f1_macro':0.4}})
        return {'status':'completed'}
    fake.fit=fake_fit;fake.infer_acquisition=fake_predict;fake.sealed_test=fake_test
    monkeypatch.setitem(sys.modules,'al_pipeline.astronomy.fusion_image.task1_training',fake)

    al.execute(root=root,initialization='randomized',seed=135,config=None,policy=None,cycle=0)
    first=al.l0_dir(root,'randomized',135)
    assert len(json.loads((first/'complete.json').read_text())['labeled_ids'])==476
    for cycle in (1,2):
        al.execute(root=root,initialization='randomized',seed=135,
                   config='random_warm',policy='entropy',cycle=cycle)
        path=al.cycle_dir(root,'random_warm','entropy',135,cycle)
        final=json.loads((path/'complete.json').read_text())
        assert len(final['selected_ids'])==473
        assert len(final['labeled_ids'])==476+473*cycle
        selection=json.loads((path/'acquisition.json').read_text())
        assert selection['label_information_available_to_selection'] is False
        assert 'target' not in selection
        # Duplicate invocation must never reselect/retrain/reveal new labels.
        al.execute(root=root,initialization='randomized',seed=135,
                   config='random_warm',policy='entropy',cycle=cycle)
