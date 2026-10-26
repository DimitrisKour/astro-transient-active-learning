"""Synthetic, CPU-only acceptance and stage-order tests with fake AstroMiNN imports."""
import importlib.util
import json
import sys
import types
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]/'src'


@pytest.fixture
def modules(monkeypatch):
    def install(name, attrs):
        mod=types.ModuleType(name)
        for k,v in attrs.items():setattr(mod,k,v)
        monkeypatch.setitem(sys.modules,name,mod)
        return mod
    install('al_pipeline.astronomy.experiments.simulated',{'SimulatedActiveLearningExperiment':type('Base',(),{})})
    install('al_pipeline.core.training.base',{'TrainingResult':type('TrainingResult',(),{}),
                                              'TrainingRequest':type('TrainingRequest',(),{})})
    install('al_pipeline.core.evaluation.base',{'ModelEvaluationResult':type('M',(),{}),'SplitEvaluation':type('S',(),{})})
    install('al_pipeline.astronomy.astrominn.training',{'AstroMiNNViTTrainer':type('Trainer',(),{})})
    def load(name,rel):
        spec=importlib.util.spec_from_file_location(name,ROOT/rel)
        module=importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules,name,module)
        spec.loader.exec_module(module)
        return module
    core=load('al_pipeline.core.analysis.retention_gate','al_pipeline/core/analysis/retention_gate.py')
    retention=load('al_pipeline.astronomy.experiments.retention','al_pipeline/astronomy/experiments/retention.py')
    retry=load('al_pipeline.astronomy.experiments.retention_retry','al_pipeline/astronomy/experiments/retention_retry.py')
    return core,retention,retry


def evaluation(mod,path,ckpt,pred):
    path.mkdir(parents=True,exist_ok=True)
    y=np.repeat(np.arange(5),20)
    v=np.full((len(y),5),-10.0)
    v[np.arange(len(y)),pred]=10
    np.savez_compressed(path/'validation_predictions.npz',object_ids=np.asarray([f'x{i}' for i in range(len(y))]),
                        object_targets=y,object_values=v)
    (path/'evaluation_summary.json').write_text(json.dumps({'checkpoint_path':str(ckpt),
       'checkpoint_sha256':mod.digest(ckpt),'model_class_names':list(mod.CLASSES),
       'splits':[{'split':'validation'}]}))


def exp_fixture(tmp_path, modules, poor=True):
    core,ret,retry=modules
    exp=retry.RetryRetentionExperiment.__new__(retry.RetryRetentionExperiment)
    exp.reference_persistent=tmp_path/'P';exp.reference_data=tmp_path/'D'
    exp.run_dir=tmp_path/'run';exp.artifact_dir=tmp_path/'artifacts'
    exp.baseline_checkpoint=tmp_path/'baseline.pth';exp.baseline_checkpoint.write_bytes(b'incumbent')
    exp.rule=core.RetentionRule()
    exp.config=types.SimpleNamespace(seed=135,strategy='entropy',max_cycles=9)
    good=np.repeat(np.arange(5),20)
    bad=good.copy();bad[20:35]=0
    evaluation(ret,exp.reference_heavy()/'cycle_000'/'evaluation',exp.baseline_checkpoint,good)
    base=tmp_path/'candidate.pth';base.write_bytes(b'candidate')
    e=tmp_path/'base_eval';evaluation(ret,e,base,bad if poor else good)
    manifest=tmp_path/'manifest.csv';manifest.write_text('object_id,label\nx0,TDE\n')
    src=exp.run_dir/'cycle_001';src.mkdir(parents=True)
    (src/'cycle_summary.json').write_text(json.dumps({'trained_checkpoint':str(base),
      'evaluation_summary':str(e/'evaluation_summary.json'),'cumulative_manifest':str(manifest),
      'selected_ids':['x0']}))
    return exp,good,bad


def test_base_and_first_retry_accepted(tmp_path,modules):
    core,ret,retry=modules
    exp,good,bad=exp_fixture(tmp_path,modules)
    b=exp._base_record(1)
    assert b['comparison']['accepted'] is False
    p=tmp_path/'retry_ckpt.pth';p.write_bytes(b'retry1')
    ev=tmp_path/'retry_eval';evaluation(ret,ev,p,good)
    compare=exp._comparison(1,p,ev)
    (tmp_path/'t_summary.json').write_text('{}')
    assert compare['accepted']
    row={'schema':retry.SCHEMA,'cycle':1,'retry_number':1,'retry_seed':1001,
         'checkpoint':str(p),'checkpoint_sha256':ret.digest(p),
         'evaluation_dir':str(ev),'evaluation_sha256':ret.digest(ev/'evaluation_summary.json'),
         'training_summary':str(tmp_path/'t_summary.json'),
         'training_summary_sha256':ret.digest(tmp_path/'t_summary.json') if (tmp_path/'t_summary.json').exists() else '',
         'initial_checkpoint_sha256':ret.digest(exp.baseline_checkpoint),
         'training_manifest_sha256':b['training_manifest_sha256'],'comparison':compare}
    ret.atomic_json(exp._retry_path(1,1),row)
    final=exp._finalize(1,row,True,1)
    assert final['accepted'] and final['accepted_retry_seed']==1001
    assert exp._incumbent(2)[0]==p
    exp.initialize=lambda:types.SimpleNamespace(last_completed_cycle=1,active_cycle=None)
    skipped=exp.retry_stage(1,2)
    assert skipped['skipped']=='earlier_retry_accepted'
    assert not exp._retry_path(1,2).exists()


def test_four_failed_retries_retain_previous_checkpoint(tmp_path,modules):
    _,ret,retry=modules
    exp,good,bad=exp_fixture(tmp_path,modules)
    base=exp._base_record(1)
    for number in range(1,5):
        p=tmp_path/f'retry{number}.pth';p.write_bytes(f'retry{number}'.encode())
        ev=tmp_path/f'e{number}';evaluation(ret,ev,p,bad)
        (tmp_path/'t_summary.json').write_text('{}')
        row={'schema':retry.SCHEMA,'cycle':1,'retry_number':number,'retry_seed':retry.RETRY_SEEDS[number-1],
             'checkpoint':str(p),'checkpoint_sha256':ret.digest(p),
             'evaluation_dir':str(ev),'evaluation_sha256':ret.digest(ev/'evaluation_summary.json'),
         'training_summary':str(tmp_path/'t_summary.json'),
         'training_summary_sha256':ret.digest(tmp_path/'t_summary.json') if (tmp_path/'t_summary.json').exists() else '',
         'initial_checkpoint_sha256':ret.digest(exp.baseline_checkpoint),
             'training_manifest_sha256':base['training_manifest_sha256'],
             'comparison':exp._comparison(1,p,ev)}
        ret.atomic_json(exp._retry_path(1,number),row)
    exp.initialize=lambda:types.SimpleNamespace(last_completed_cycle=1,active_cycle=None)
    out=exp.retry_stage(1,4)
    assert not out['comparison']['accepted']
    decision=exp.check_final(1)
    assert not decision['accepted'] and decision['attempts_consumed']==4
    assert decision['incumbent_checkpoint']==str(exp.baseline_checkpoint)
    assert exp._incumbent(2)[0]==exp.baseline_checkpoint


def test_no_premature_retry_and_tampered_source(tmp_path,modules):
    _,ret,retry=modules
    exp,good,bad=exp_fixture(tmp_path,modules)
    exp._base_record(1)
    exp.initialize=lambda:types.SimpleNamespace(last_completed_cycle=1,active_cycle=None)
    with pytest.raises(RuntimeError,match='Earlier retry missing'):
        exp.retry_stage(1,2)
    exp.baseline_checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='provenance|mismatch'):
        exp._base_record(1)


def test_accept_base_stops_retries(tmp_path,modules):
    _,ret,retry=modules
    exp,good,bad=exp_fixture(tmp_path,modules,poor=False)
    base=exp._base_record(1)
    assert base['comparison']['accepted']
    exp._finalize(1,base,True,None)
    exp.initialize=lambda:types.SimpleNamespace(last_completed_cycle=1,active_cycle=None)
    assert exp.retry_stage(1,1)['skipped']=='base_accepted'


def test_retry_trains_with_same_manifest_and_parent_and_earliest_seed(tmp_path,modules,monkeypatch):
    from dataclasses import dataclass
    _,ret,retry=modules
    exp,good,bad=exp_fixture(tmp_path,modules)
    base=exp._base_record(1)
    @dataclass(frozen=True)
    class Policy:
        loader_seed:int=135
        class_weight_exponent:float=.85
        new_object_weighting:str='per_sample'
        class_weight_source:str='training_manifest'
        overwrite:bool=False
    class Request:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    class FakeTrainer:
        calls=[]
        def __init__(self,**kwargs):
            self.policy=kwargs['policy'];self.kwargs=kwargs
        def completed_result(self,request):return None
        def train(self,request):
            assert request.seed==1001 and self.policy.loader_seed==1001
            assert request.initial_checkpoint==exp.baseline_checkpoint
            assert request.train_manifest==Path(base['training_manifest'])
            assert request.validation_manifest==exp.dataset.validation_manifest
            self.calls.append(request)
            request.output_dir.mkdir(parents=True,exist_ok=True)
            p=request.output_dir/'best_vit_tower.pth';p.write_bytes(b'fresh_retry1')
            summary=request.output_dir/'training_summary.json';summary.write_text('{}')
            return types.SimpleNamespace(checkpoint_path=p,summary_path=summary,elapsed_seconds=3.5)
    class FakeEvaluator:
        def completed_result(self,**kwargs):return None
        def evaluate(self,**kwargs):
            evaluation(ret,kwargs['output_dir'],kwargs['checkpoint_path'],good)
            return types.SimpleNamespace(summary_path=kwargs['output_dir']/'evaluation_summary.json')
    tr=types.SimpleNamespace(policy=Policy(),device='cuda',initialization='pretrained',
            legacy_code_dir=tmp_path,artifact_dir=tmp_path,model_factory=None)
    exp.trainer=tr
    exp.dataset=types.SimpleNamespace(validation_manifest=tmp_path/'validation.csv')
    exp.dataset.validation_manifest.write_text('obj,label\n')
    exp.evaluator=FakeEvaluator()
    exp.config.run_id='synthetic_retry'
    exp.initialize=lambda:types.SimpleNamespace(last_completed_cycle=1,active_cycle=None)
    monkeypatch.setattr(retry,'TrainingRequest',Request)
    monkeypatch.setattr(retry,'AstroMiNNViTTrainer',FakeTrainer)
    result=exp.retry_stage(1,1)
    assert result['comparison']['accepted'] and len(FakeTrainer.calls)==1
    assert exp.check_final(1)['accepted_retry_seed']==1001
    assert exp.retry_stage(1,2)['skipped']=='earlier_retry_accepted'
    assert len(FakeTrainer.calls)==1
