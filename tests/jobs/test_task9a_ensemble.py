import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research/historical_replay/task9'))
sys.path.insert(0,str(ROOT/'src'))
from task9a_ensemble import VARIANTS,assemble,model_stats,paired_boot,count_changes,ece

def _sample(p):
    y=np.array([0,1,2,3,4]);return np.array(['a','b','c','d','e']),y,np.asarray(p,dtype=float)

def test_fixed_members_are_not_best_of_five():
    inc=_sample(np.eye(5))
    tries=[_sample(np.full((5,5),i/100)+np.eye(5)/5) for i in range(1,6)]
    assert VARIANTS==('E2','E4','I_E2','I_E4','E5','I_E5')
    for label,n,include in [('E2',2,0),('E4',4,0),('I_E2',2,1),('I_E4',4,1),('E5',5,0),('I_E5',5,1)]:
        x=assemble(inc,tries,label)[2]
        expected=np.mean([q[2] for q in tries[:n]]+([inc[2]] if include else []),axis=0)
        np.testing.assert_allclose(x,expected)

def test_probability_metrics_and_paired_bootstrap():
    y=np.tile(np.arange(5),3)
    p=np.eye(5)[y]*.8+.04
    metric=model_stats(y,p)
    assert metric['macro_f1']==pytest.approx(1)
    assert metric['balanced_accuracy']==pytest.approx(1)
    assert metric['ece']==pytest.approx(.16)
    assert count_changes(y,p,p)==(0,0,0)
    bounds=paired_boot(y,p,p,seed=1,n=20)
    assert bounds['macro_f1']==pytest.approx((0.,0.))
    assert bounds['balanced_accuracy']==pytest.approx((0.,0.))

def test_full_report_synthetic_six_cases(monkeypatch,tmp_path):
    import task9a_ensemble as m
    import json
    plan={'schema':'task8b_first_rejection_v1','cases':{
        f'entropy_s{seed}':{'case':{'policy':'entropy','seed':seed,'rejected_cycle':4}} for seed in (135,246,357)}|
        {f'entropy_balanced_explore_s{seed}':{'case':{'policy':'entropy_balanced_explore','seed':seed,'rejected_cycle':4}} for seed in (135,246,357)},
        'jobs':[{}]*30,'source_audit':str(tmp_path/'audit.csv'),'source_audit_sha256':'mock'}
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(m,'sha',lambda *a:'mock')
    y=np.tile(np.arange(5),3);ids=np.array([str(i) for i in range(len(y))]);p=np.eye(5)[y]*.8+.04
    sample=(ids,y,p)
    p_alt=p.copy();p_alt[0]=np.roll(p_alt[0],1)
    different=(ids,y,p_alt)
    monkeypatch.setattr(m,'verified_case',lambda plan,key:(sample,sample,[different]*5,plan['cases'][key]['case']))
    result=m.report(tmp_path,tmp_path/'out',bootstrap=4)
    assert result['cases']==6 and result['comparisons']==6*6*2
    for name in ('ensemble_comparisons.csv','class_recall.csv','prediction_flips.csv','confusions.csv','summary.json'):
        assert (tmp_path/'out'/name).is_file()

def test_validation_reader_checks_checkpoint_and_aligns(tmp_path):
    import json
    import task9_predictions as m
    cp=tmp_path/'checkpoint.pth';cp.write_bytes(b'checkpoint')
    fold=tmp_path/'evaluation';fold.mkdir()
    meta={'checkpoint_path':str(cp),'checkpoint_sha256':m.sha256(cp),
          'model_class_names':list(m.CLASSES),'splits':[{'split':'validation'}]}
    (fold/'evaluation_summary.json').write_text(json.dumps(meta))
    ids=np.array(['b','a']);targets=np.array([1,0])
    logits=np.array([[0.,10.,0.,0.,0.],[10.,0.,0.,0.,0.]])
    np.savez(fold/'validation_predictions.npz',object_ids=ids,object_targets=targets,object_values=logits)
    out=m.load_prediction(fold/'validation_predictions.npz',cp)
    assert out[0].tolist()==['a','b']
    assert out[1].tolist()==[0,1]
    assert out[2].argmax(1).tolist()==[0,1]
    with pytest.raises(ValueError,match='only read validation'):
        m.load_prediction(fold/'test_predictions.npz',cp)
    cp.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA256 changed'):
        m.load_prediction(fold/'validation_predictions.npz',cp)
