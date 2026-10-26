"""Synthetic provenance/retention tests without the heavy frozen AstroMiNN package."""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    # Stub imported interfaces; production installation takes the actual frozen source.
    for name, attrs in {
        'al_pipeline.astronomy.experiments.simulated':{'SimulatedActiveLearningExperiment':type('Base',(),{})},
        'al_pipeline.core.training.base':{'TrainingResult':type('TrainingResult',(),{})},
        'al_pipeline.core.evaluation.base':{'ModelEvaluationResult':type('ModelEvaluationResult',(),{}),
                                            'SplitEvaluation':type('SplitEvaluation',(),{})},
    }.items():
        fake=types.ModuleType(name)
        for key,value in attrs.items():setattr(fake,key,value)
        monkeypatch.setitem(sys.modules,name,fake)
    path=Path(__file__).resolve().parents[2]/'src/al_pipeline/astronomy/experiments/retention.py'
    spec=importlib.util.spec_from_file_location('task8c1_synthetic_retention',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluation(mod,path,checkpoint,y,pred):
    path.mkdir(parents=True,exist_ok=True)
    v=np.full((len(y),5),-10.0)
    v[np.arange(len(y)),pred]=10
    np.savez_compressed(path/'validation_predictions.npz',
         object_ids=np.asarray([f'x{i}' for i in range(len(y))]),
         object_targets=y,object_values=v)
    (path/'evaluation_summary.json').write_text(json.dumps({
        'checkpoint_path':str(checkpoint),'checkpoint_sha256':mod.digest(checkpoint),
        'model_class_names':list(mod.CLASSES),
        'splits':[{'split':'validation'}]}))


def synthetic(tmp_path,rejected,mod):
    P=tmp_path/'P';D=tmp_path/'D';run=tmp_path/'run'
    exp=mod.RetentionExperiment.__new__(mod.RetentionExperiment)
    exp.reference_persistent=P;exp.reference_data=D;exp.run_dir=run
    exp.rule=mod.RetentionRule()
    exp.config=types.SimpleNamespace(seed=135,strategy='entropy')
    exp.baseline_checkpoint=tmp_path/'baseline.pth'
    exp.baseline_checkpoint.write_bytes(b'base')
    new=tmp_path/'candidate.pth';new.write_bytes(b'new')
    y=np.repeat(np.arange(5),20);pred=y.copy()
    if rejected:pred[20:30]=0
    evaluation(mod,D/'runs'/mod.REFERENCE_SUITE/'entropy'/'seed_135'/'cycle_000'/'evaluation',
               exp.baseline_checkpoint,y,y)
    evaldir=tmp_path/'evaluation'
    evaluation(mod,evaldir,new,y,pred)
    manifest=tmp_path/'manifest.csv';manifest.write_text('id,label\nobj1,TDE\n')
    src=run/'cycle_001';src.mkdir(parents=True)
    (src/'cycle_summary.json').write_text(json.dumps({
        'trained_checkpoint':str(new),'evaluation_summary':str(evaldir/'evaluation_summary.json'),
        'cumulative_manifest':str(manifest),'selected_ids':['obj1']}))
    return mod,exp,new


def test_rejection_retains_same_parent_and_data(tmp_path,module):
    mod,exp,_=synthetic(tmp_path,True,module)
    result=exp._finish_decision(1)
    assert not result['accepted']
    assert result['incumbent_checkpoint']==str(exp.baseline_checkpoint)
    assert result['training_manifest_sha256'] == mod.digest(tmp_path/'manifest.csv')
    assert result['incumbent_source_cycle']==0
    assert exp._incumbent(2)[0]==exp.baseline_checkpoint
    assert exp._finish_decision(1)==result


def test_acceptance_promotes_candidate(tmp_path,module):
    mod,exp,new=synthetic(tmp_path,False,module)
    result=exp._finish_decision(1)
    assert result['accepted']
    assert result['incumbent_checkpoint']==str(new)
    assert exp._incumbent(2)[0]==new


def test_changed_reference_model_refused(tmp_path,module):
    mod,exp,_=synthetic(tmp_path,False,module)
    exp._finish_decision(1)
    exp.baseline_checkpoint.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='provenance|mismatch'):
        exp._finish_decision(1)


def test_changing_decision_refused(tmp_path,module):
    mod,exp,_=synthetic(tmp_path,False,module)
    exp._finish_decision(1)
    p=exp._decision_path(1);row=json.loads(p.read_text());row['accepted']=False
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='does not match recalculation'):
        exp._finish_decision(1)
