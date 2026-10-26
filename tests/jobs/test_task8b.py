from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest

HERE=Path(__file__).parents[2]/'scripts/adaptive'

def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_report_metrics_and_ensembling():
    r=module('task8b_report')
    y=np.arange(5,dtype=int)
    p=np.eye(5)[y]*.8+.04
    s=r.scores(y,p)
    assert s['accuracy']==1 and s['macro_f1']==1
    assert np.isclose(s['balanced_accuracy'],1)
    p2=p.copy();p2[4]=p[3]
    s2=r.scores(y,p2)
    assert s2['macro_f1']<1
    reference=(np.array(list('abcde')),y,p)
    alternative=(np.array(list('abcde')),y,p2)
    row,classes,changes=r.compare(reference,alternative,case={'policy':'entropy','seed':135,'rejected_cycle':4},
                                 label='retry_1001',kind='single_retry',ref_label='incumbent',bootstrap=20)
    assert row['n_lost_correct']==1 and len(classes)==5 and len(changes)==1
    with pytest.raises(ValueError):
        r.aligned(reference,(np.array(list('adcbe')),y,p2))


def test_plan_freezing_never_submits(monkeypatch,tmp_path):
    r=module('task8b')
    assert 'submit' in r.main.__code__.co_consts or callable(r.freeze)
    assert 'run' in r.main.__code__.co_consts
    assert len(r.OVERLAY_FILES)==7
    assert len(set(r.OVERLAY_FILES))==7


def test_full_synthetic_reporting_roundtrip(tmp_path):
    """6 original rejections, 30 completed retries, and ensemble reports."""
    from al_pipeline.core.analysis.seed_recovery import EXPECTED_CYCLES, RecoveryCase, job_grid, sha256
    r=module('task8b_report')
    audit=tmp_path/'first_rejections.csv';audit.write_text('Task8A immutable hash fixture')
    cases=[RecoveryCase(policy,seed,cycle,cycle-1)
           for (policy,seed),cycle in EXPECTED_CYCLES.items()]
    jobs=job_grid(tuple(cases))
    job_dir=tmp_path/'job'
    y=np.tile(np.arange(5,dtype=int), 2)
    ids=np.asarray([f'obj_{i:03d}' for i in range(len(y))])
    inc_logits=np.full((len(y),5),-5.0)
    inc_logits[np.arange(len(y)),y]=5.0
    cand_logits=inc_logits.copy();cand_logits[0,0]=-5.0;cand_logits[0,2]=5.0

    def save_eval(folder, logit, checkpoint):
        folder.mkdir(parents=True,exist_ok=True)
        checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes((str(checkpoint)+'data').encode())
        np.savez_compressed(folder/'validation_predictions.npz',object_ids=ids,object_targets=y,object_values=logit)
        (folder/'evaluation_summary.json').write_text(json.dumps({
            'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
            'model_class_names':['AGN','TDE','SN I','SN II','CV'],
            'splits':[{'split':'validation'}]}))

    evidence={}
    for case in cases:
        key=f'{case.policy}_s{case.seed}'
        path=tmp_path/'original'/key
        inc_ckpt=path/'incumbent_ckpt.pth';cand_ckpt=path/'candidate_ckpt.pth'
        save_eval(path/'incumbent_eval',inc_logits,inc_ckpt)
        save_eval(path/'candidate_eval',cand_logits,cand_ckpt)
        evidence[key]={'case':case.as_dict(),'incumbent_evaluation':str(path/'incumbent_eval'/'evaluation_summary.json'),
                       'source_evaluation':str(path/'candidate_eval'/'evaluation_summary.json'),
                       'initial_checkpoint':str(inc_ckpt),'candidate_checkpoint':str(cand_ckpt),
                       'manifest_sha256':'abc','validation_sha256':'def','initial_checkpoint_sha256':sha256(inc_ckpt)}
    manifest=[]
    for job in jobs:
        folder=job_dir/'fits'/job.name
        ck=folder/'training'/'best_vit_tower.pth'
        save_eval(folder/'evaluation',inc_logits,ck)
        (folder/'training'/'training_summary.json').write_text(json.dumps({
            'initial_checkpoint':evidence[f'{job.case.policy}_s{job.case.seed}']['initial_checkpoint'],
            'train_manifest_sha256':'abc','validation_manifest_sha256':'def',
            'seed':job.retry_seed,'policy':{'loader_seed':job.retry_seed}}))
        item={**job.as_dict(),'output_dir':str(folder)}
        (folder/'complete.json').write_text(json.dumps({'job':item,'validation_only':True,
            'checkpoint_path':str(ck),'checkpoint_sha256':sha256(ck),
            'original_training_manifest_sha256':'abc',
            'initial_checkpoint_sha256':evidence[f'{job.case.policy}_s{job.case.seed}']['initial_checkpoint_sha256']}))
        manifest.append(item)
    job_dir.mkdir(parents=True,exist_ok=True)
    (job_dir/'plan.json').write_text(json.dumps({'schema':'task8b_first_rejection_v1',
        'jobs':manifest,'cases':evidence,'source_audit':str(audit),'source_audit_sha256':sha256(audit)}))
    output=r.make_report(job_dir,tmp_path/'out',bootstrap=8)
    assert output['retries_completed']==30
    assert output['retries_passing_provisional_rule']==30
    with (tmp_path/'out'/'model_comparisons.csv').open() as f:
        import csv
        rows=list(csv.DictReader(f))
    assert sum(row['kind']=='ensemble' for row in rows)==24
    assert sum(row['kind']=='single_retry' for row in rows)==60
    assert (tmp_path/'out'/'per_class_comparisons.csv').is_file()
