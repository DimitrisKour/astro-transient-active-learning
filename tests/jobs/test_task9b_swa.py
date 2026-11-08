import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research/historical_replay/task9'))
sys.path.insert(0,str(ROOT/'src'))
from task9b_swa import make_plan,submit,status,atomic_json
from al_pipeline.astronomy.analysis.swa_continuation import (EPOCHS,START_AVERAGING,CONTINUATION_LR,
                                                             assert_swa_architecture)

def test_fixed_science_and_bn_guard():
    import torch
    from torch.optim.swa_utils import AveragedModel
    assert (EPOCHS,START_AVERAGING,CONTINUATION_LR)==(10,6,2e-5)
    a=torch.nn.Linear(2,1,bias=False)
    b=AveragedModel(a,use_buffers=True)
    for value in (1.,2.,3.,4.,5.):
        a.weight.data.fill_(value);b.update_parameters(a)
    assert float(b.module.weight.detach().mean())==pytest.approx(3.)
    assert int(b.n_averaged)==5
    assert_swa_architecture(a)
    with pytest.raises(ValueError,match='BatchNorm'):assert_swa_architecture(torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.BatchNorm1d(1)))

def test_planning_requires_independent_output(monkeypatch,tmp_path):
    import task9b_swa
    fake={'schema':'task8b_first_rejection_v1','cases':{
        f'entropy_s{i}':{'case':{'policy':'entropy','seed':i,'rejected_cycle':4}} for i in range(6)},
        'jobs':[{} for _ in range(30)]}
    monkeypatch.setattr(task9b_swa,'check_task8b',lambda *a,**k:fake)
    monkeypatch.setattr(task9b_swa,'hash_file',lambda *a:'mockdigest')
    x=make_plan(tmp_path/'b',tmp_path/'out')
    assert len(x['cases'])==6 and x['gpu_fits']==6
    assert x['experimental_policy']['average_epochs']==[6,7,8,9,10]
    with pytest.raises(ValueError,match='independent'):make_plan(tmp_path/'b',tmp_path/'b'/'out')

def test_submission_rejects_duplicates_without_sbatch(monkeypatch,tmp_path):
    import task9b_swa
    c={'key':'case','output_dir':str(tmp_path/'x')}
    monkeypatch.setattr(task9b_swa,'verify',lambda *a,**kw:{'cases':[c]*6})
    atomic_json(tmp_path/'submitted_case_0.json',{'job_id':'12345'})
    with pytest.raises(RuntimeError,match='already queued'):submit(tmp_path,[0],'00:20:00')
    with pytest.raises(ValueError,match='distinct'):submit(tmp_path,[1,1],'00:20:00')
    with pytest.raises(ValueError,match='Walltime'):submit(tmp_path,[1],'00:25:00')

def test_swa_contract_requires_stable_source(tmp_path):
    from al_pipeline.astronomy.analysis.swa_continuation import static_contract
    p=tmp_path/'L0.csv';p.write_text('data')
    inp={'candidate_checkpoint':'cp','candidate_checkpoint_sha256':'sha',
         'training_manifest':'train','manifest_sha256':'trainsha',
         'validation_manifest':'validation','validation_sha256':'validsha',
         'l0_manifest':str(p),'source_training_summary_sha256':'reportsha',
         'model_policy':{'class_weight_exponent':.85}}
    case={'policy':'entropy','seed':135,'rejected_cycle':4}
    a=static_contract(inp,case)
    assert a['averaging_epochs']==[6,7,8,9,10]
    assert a['original_candidate_checkpoint']=='cp'
    assert a['l0_manifest_sha256']
    p.write_text('changed')
    assert static_contract(inp,case)['l0_manifest_sha256'] != a['l0_manifest_sha256']

def test_swa_report_six_case_synthetic(monkeypatch,tmp_path):
    import task9b_swa_report as m
    import json
    import numpy as np
    src={'incumbent_evaluation':'inc/evaluation_summary.json','source_evaluation':'orig/evaluation_summary.json',
         'initial_checkpoint':'inc.pth','candidate_checkpoint':'orig.pth'}
    cases=[]
    for seed in (135,246,357):
        for policy in ('entropy','entropy_balanced_explore'):
            key=f'{policy}_s{seed}';out=tmp_path/key;out.mkdir()
            case={'key':key,'policy':policy,'seed':seed,'rejected_cycle':4,
                  'output_dir':str(out),'source':src}
            cases.append(case)
            marker={'case':case,'validation_only':True,'test_used':False,
                    'training_result':{'training_seconds_this_launch':3.,'swa_members':5,
                                       'final_sha256':'test','swa_sha256':'test'},
                    'single_evaluation_summary_sha256':'test','swa_evaluation_summary_sha256':'test'}
            (out/'complete.json').write_text(json.dumps(marker))
    (tmp_path/'plan.json').write_text(json.dumps({'schema':'task9b_swa_v1','cases':cases}))
    monkeypatch.setattr(m,'sha',lambda *a:'test')
    y=np.tile(np.arange(5),3);ids=np.asarray([str(i) for i in range(len(y))]);p=np.eye(5)[y]*.8+.04
    a=(ids,y,p)
    monkeypatch.setattr(m,'load_prediction',lambda *args,**kwargs:a)
    result=m.report(tmp_path,tmp_path/'report',bootstrap=4)
    assert result['cases']==6 and result['comparisons']==30
    for file in ('comparisons.csv','class_recall.csv','prediction_flips.csv','confusions.csv','summary.json'):
        assert (tmp_path/'report'/file).is_file()
