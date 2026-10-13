"""Checkpoint lineage, identical saved data, reference reuse and interrupted fits."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import shutil
import pytest
import torch
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer, AstroMiNNViTTrainingConfig
from al_pipeline.astronomy.initialization_compare import prepare_comparison, run_task, batch_selection
from al_pipeline.astronomy.initialization_reporting import report_comparison
from al_pipeline.astronomy.training_diagnostics import ORDER_SEEDS, ORDER_STRATEGIES, load_plan, sha256, verify_plan, write_json
from al_pipeline.core.training.base import TrainingResult
from al_pipeline.core.evaluation.base import ModelEvaluationResult, SplitEvaluation

ROOT=Path(__file__).resolve().parents[2]

def manifest(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['obj_id','filepath','target','alert','survey']);w.writeheader();w.writerows(rows)

@pytest.fixture
def inputs(tmp_path):
    base=tmp_path/'input';base.mkdir();legacy=base/'legacy';legacy.mkdir()
    (legacy/'astroMiNN_V2.py').write_text('# runtime model')
    model=base/'model';model.mkdir()
    for name in ('astroMiNN_config.json','vit_tower_135.pth','best_lc_tower_135.pth','best_spatial_tower_135.pth','best_coord_tower_135.pth'):
        (model/name).write_text('{}')
    rows=[]
    for i in range(12):
        p=base/f'sample_{i}.npy';p.write_bytes(b'fixture')
        rows.append({'obj_id':f'obj_{i:02d}','filepath':str(p),'target':['AGN','CV','SN I','SN II','TDE'][i%5],'alert':'1','survey':'ZTF'})
    rows.append(dict(rows[0]))
    full=base/'train.csv';val=base/'val.csv';test=base/'test.csv'
    manifest(full,rows);manifest(val,[{**rows[0],'obj_id':'validation'}]);manifest(test,[{**rows[0],'obj_id':'test'}])
    persistent=tmp_path/'persistent';heavy=tmp_path/'heavy';registry=base/'registry.toml'
    registry.write_text(f'''[storage]\npersistent_root="{persistent}"\ndata_root="{heavy}"\n[datasets.astrominn_ztf]\nsurvey="ZTF"\nmodality="image_metadata"\nselection_unit="object"\ntrain_manifest="{full}"\nvalidation_manifest="{val}"\ntest_manifest="{test}"\ndata_root="{base}"\nobject_id_column="obj_id"\nsample_file_column="filepath"\nlabel_column="target"\n[profiles.astrominn_ztf]\nmodel="AstroMiNN"\ndataset="astrominn_ztf"\ncheckpoint="{model/'vit_tower_135.pth'}"\n''')
    suite=persistent/'runs/astrominn_historical_replay_v1';artifacts=heavy/'runs/astrominn_historical_replay_v1'
    l0=suite/'shared/historical_replay_split/initial_labeled_manifest.csv'
    l0rows=[r for r in rows if int(r['obj_id'][-2:])<3];manifest(l0,l0rows)
    for seed in ORDER_SEEDS:
        policy=asdict(AstroMiNNViTTrainingConfig(loader_seed=seed,max_epochs=2,patience=1,train_num_workers=0,validation_num_workers=0,class_weight_source='training_manifest'))
        policy=json.loads(json.dumps(policy))
        bootstrap_dir=artifacts/'shared'/f'seed_{seed}'/'bootstrap_training';bootstrap_dir.mkdir(parents=True)
        checkpoint=bootstrap_dir/'best_vit_tower.pth';torch.manual_seed(seed);torch.save(torch.nn.Linear(2,2).state_dict(),checkpoint)
        bs={'cycle':0,'strategy':'bootstrap','seed':seed,'initialization':'random_vit','initial_checkpoint':None,
            'train_manifest_sha256':sha256(l0),'validation_manifest_sha256':sha256(val),'training_samples':len(l0rows),'policy':policy,
            'epochs_completed':2,'best_epoch':1,'elapsed_seconds':2}
        write_json(bootstrap_dir/'training_summary.json',bs)
        for n,strategy in enumerate(ORDER_STRATEGIES):
            state=suite/strategy/f'seed_{seed}';state.mkdir(parents=True)
            write_json(state/'run_state.json',{'last_completed_cycle':9,'active_cycle':None})
            write_json(state/'experiment.json',{'config':{'training_initialization':'restart_from_random'},
                'artifact_dir':str(artifacts/strategy/f'seed_{seed}'),'validation_manifest_sha256':sha256(val),
                'historical_train_manifest':str(l0),'historical_train_manifest_sha256':sha256(l0),
                'baseline_checkpoint':str(checkpoint),'baseline_checkpoint_sha256':sha256(checkpoint)})
            history=[]
            for cycle in range(1,10):
                chosen=[r for r in rows if int(r['obj_id'][-2:])<cycle+3];chosen=chosen[n:]+chosen[:n]
                path=state/'manifests'/f'train_cycle_{cycle:03d}.csv';manifest(path,chosen)
                summary={**bs,'cycle':cycle,'strategy':strategy,'training_samples':len(chosen),'train_manifest_sha256':sha256(path)}
                write_json(artifacts/strategy/f'seed_{seed}'/f'cycle_{cycle:03d}/training/training_summary.json',summary)
                metrics={'f1_macro':.6,'accuracy':.7,'balanced_accuracy':.65}
                evaluation=[{'split':s,'sample_count':1,'object_count':1,'sample_metrics':metrics,'object_metrics':metrics} for s in ('validation','test')]
                write_json(artifacts/strategy/f'seed_{seed}'/f'cycle_{cycle:03d}/evaluation/evaluation_summary.json',{'splits':evaluation})
                history.extend({'cycle':cycle,'split':s,'metrics':metrics} for s in ('validation','test'))
            write_json(state/'evaluation_history.json',history)
    return {'repo':ROOT,'registry_path':registry,'legacy_code_dir':legacy,'comparison_id':'test_init'}


def test_default_204_new_fits_reuse_random_and_share_first_cycle(inputs):
    path=prepare_comparison(**inputs);plan=load_plan(path)
    assert len(plan['tasks'])==204 and len(plan['references'])==108
    assert len(plan['bootstraps'])==3
    assert sum(t['cycle']==1 for t in plan['tasks'])==12
    assert all(t['modes']==['restart_from_l0','warm_start'] for t in plan['tasks'] if t['cycle']==1)
    for t in plan['tasks']:
        if t['mode']=='warm_start':
            previous=next(x for x in plan['tasks'] if x['task_id']==t['initial_task'])
            assert (previous['seed'],previous['policy_name'],previous['cycle'])==(t['seed'],t['policy_name'],t['cycle']-1)
        else: assert t['initial_checkpoint']==plan['bootstraps'][str(t['seed'])]['checkpoint']
        ref=next(r for r in plan['references'] if (r['seed'],r['policy'],r['cycle'])==(t['seed'],t['policy_name'],t['cycle']))
        assert t['manifest_sha256']==ref['manifest_sha256']
    assert prepare_comparison(**inputs)==path
    assert not list(Path(plan['artifact_root']).glob('s*'))


def test_wrong_bootstrap_hash_rejected_before_writes(inputs):
    root=inputs['registry_path'].parent.parent/'heavy/runs/astrominn_historical_replay_v1'
    (root/'shared/seed_135/bootstrap_training/best_vit_tower.pth').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bootstrap'): prepare_comparison(**inputs)
    assert not (inputs['registry_path'].parent.parent/'persistent/initialization_compare').exists()


def test_changed_class_and_missing_existing_rows_rejected(inputs):
    root=inputs['registry_path'].parent.parent/'persistent/runs/astrominn_historical_replay_v1'
    p=root/'random/seed_135/manifests/train_cycle_002.csv'
    p.write_text(p.read_text().replace('AGN','CV'))
    summary=inputs['registry_path'].parent.parent/'heavy/runs/astrominn_historical_replay_v1/random/seed_135/cycle_002/training/training_summary.json'
    d=json.loads(summary.read_text());d['train_manifest_sha256']=sha256(p);write_json(summary,d)
    with pytest.raises(ValueError,match='removed or changed'):prepare_comparison(**inputs)


def test_external_wandb_symlinks_ignored_and_plan_frozen(inputs):
    p=inputs['legacy_code_dir']/'wandb/old';p.mkdir(parents=True);(p/'dangling.py').symlink_to(p/'absent')
    path=prepare_comparison(**inputs);plan=load_plan(path)
    assert not any('dangling' in k for k in plan['immutable_inputs'])
    task=plan['tasks'][0];m=Path(task['manifest']);m.write_text(m.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):verify_plan(path)


class FakeTrainer(AstroMiNNViTTrainer):
    calls=[]
    rng_after_model=[]
    fail_once=False
    def train(self,request):
        self._validate_request(request);self._seed_process(request.seed)
        assert request.initial_checkpoint is not None
        model=self._build_model(request)
        self.rng_after_model.append(torch.get_rng_state().clone())
        self.calls.append(request)
        request.output_dir.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():model.vit_tower.weight.add_(.1)
        checkpoint=request.output_dir/'best_vit_tower.pth';torch.save(model.vit_tower.state_dict(),checkpoint)
        if type(self).fail_once:
            type(self).fail_once=False;raise RuntimeError('interrupted training')
        history=request.output_dir/'training_history.json';history.write_text('[]')
        summary=request.output_dir/'training_summary.json'
        payload={'run_id':request.run_id,'cycle':request.cycle,'strategy':request.strategy,'seed':request.seed,
          'initialization':self.initialization,'initial_checkpoint':str(request.initial_checkpoint),'policy':asdict(self.policy),
          'train_manifest_sha256':sha256(request.train_manifest),'validation_manifest_sha256':sha256(request.validation_manifest),
          'checkpoint_path':str(checkpoint),'history_path':str(history),'best_epoch':1,'epochs_completed':1,'best_validation_loss':.1,'elapsed_seconds':1}
        write_json(summary,payload)
        return TrainingResult(checkpoint,history,summary,1,1,.1,1)

class FakeEvaluator:
    fail_once=False
    def __init__(self,**kwargs):pass
    def evaluate(self,*,checkpoint_path,output_dir):
        if type(self).fail_once:
            type(self).fail_once=False;raise RuntimeError('interrupted evaluation')
        output_dir.mkdir(parents=True,exist_ok=True)
        metrics={'f1_macro':.62,'accuracy':.72,'balanced_accuracy':.67}
        return ModelEvaluationResult(checkpoint_path,output_dir/'summary.json',tuple(SplitEvaluation(s,1,1,metrics,metrics) for s in ('validation','test')))

def toy_loader(**kwargs):
    assert kwargs['load_vit_checkpoint']
    model=torch.nn.Linear(2,2);model.load_state_dict(torch.load(kwargs['vit_checkpoint'],weights_only=True))
    return SimpleNamespace(vit_tower=model)

def execute(path,index):
    return run_task(path,index,device='cpu',trainer_class=FakeTrainer,evaluator_class=FakeEvaluator,model_loader=toy_loader)

@pytest.fixture(autouse=True)
def reset():
    FakeTrainer.calls=[];FakeTrainer.rng_after_model=[];FakeTrainer.fail_once=False;FakeEvaluator.fail_once=False


def test_warm_uses_predecessor_l0_restart_uses_bootstrap_and_shared_cycle1(inputs):
    inputs.update(seeds=(135,),max_cycle=2)
    path=prepare_comparison(**inputs);plan=load_plan(path);first=plan['tasks'][0]
    warm=next(t for t in plan['tasks'] if t['policy_name']=='random' and t['mode']=='warm_start')
    restart=next(t for t in plan['tasks'] if t['policy_name']=='random' and t['mode']=='restart_from_l0')
    with pytest.raises(ValueError,match='predecessor'):execute(path,warm['index'])
    initial=execute(path,first['index']);execute(path,warm['index']);execute(path,restart['index'])
    first_result=json.loads(initial.read_text())
    assert FakeTrainer.calls[1].initial_checkpoint==Path(first_result['checkpoint_path'])
    assert FakeTrainer.calls[2].initial_checkpoint==Path(first['initial_checkpoint'])
    assert len(FakeTrainer.calls)==3
    execute(path,first['index']);assert len(FakeTrainer.calls)==3
    report=report_comparison(path,path.parent/'report',plots=False)
    assert report['completed_new_fits']==3 and report['new_training_seconds_completed']==3
    rows=list(csv.DictReader((path.parent/'report/paired_differences.csv').open()))
    shared=[r for r in rows if r['cycle']=='1' and r['policy']=='random' and r['comparison']=='warm_start minus restart_from_l0']
    assert shared and all(float(r['delta_pp'])==0 for r in shared)


def test_completed_training_resumes_evaluation_without_retraining(inputs):
    inputs.update(seeds=(135,),max_cycle=1);path=prepare_comparison(**inputs)
    FakeEvaluator.fail_once=True
    with pytest.raises(RuntimeError,match='evaluation'):execute(path,0)
    execute(path,0);assert len(FakeTrainer.calls)==1
    assert 0 not in batch_selection(path,135,1)


def test_interrupted_training_preserved_in_new_attempt(inputs):
    inputs.update(seeds=(135,),max_cycle=1);path=prepare_comparison(**inputs)
    FakeTrainer.fail_once=True
    with pytest.raises(RuntimeError,match='training'):execute(path,0)
    result=json.loads(execute(path,0).read_text())
    assert 'attempt_002' in result['checkpoint_path'] and len(FakeTrainer.calls)==2


def test_report_portable_partial_counts_and_lineage_rejection(inputs,tmp_path):
    inputs.update(seeds=(135,),max_cycle=2);path=prepare_comparison(**inputs)
    execute(path,0)
    portable=tmp_path/'download';portable.mkdir()
    for name in ('plan.json','plan.sha256'):shutil.copyfile(path.parent/name,portable/name)
    shutil.copytree(path.parent/'tasks',portable/'tasks')
    report=report_comparison(portable/'plan.json',portable/'report',plots=False)
    assert report['completed_new_fits']==1 and len(report['incomplete'])==11
    first=load_plan(path)['tasks'][0]
    p=portable/'tasks'/first['task_id']/'result.json';d=json.loads(p.read_text());d['initialization']['checkpoint_sha256']='wrong';write_json(p,d)
    with pytest.raises(ValueError,match='lineage'):report_comparison(portable/'plan.json',portable/'report2',plots=False)


def test_initialization_audit_does_not_consume_rng_or_change_checkpoint(inputs):
    inputs.update(seeds=(135,),max_cycle=1);path=prepare_comparison(**inputs);plan=load_plan(path)
    checkpoint=Path(plan['tasks'][0]['initial_checkpoint']);before=sha256(checkpoint)
    observed={}
    def loader(**kwargs):
        model=toy_loader(**kwargs);observed['rng']=torch.get_rng_state().clone();return model
    run_task(path,0,device='cpu',trainer_class=FakeTrainer,evaluator_class=FakeEvaluator,model_loader=loader)
    assert torch.equal(observed['rng'],FakeTrainer.rng_after_model[0])
    init=json.loads((path.parent/'tasks'/plan['tasks'][0]['task_id']/'initialization.json').read_text())
    assert init['loaded_vit_checkpoint'] and init['optimizer_reset'] and init['scheduler_reset']
    assert sha256(checkpoint)==before


def test_complete_two_seed_trajectories_report_paired_gains_and_no_missing_cycles(inputs):
    inputs.update(seeds=(135,246))
    path=prepare_comparison(**inputs);plan=load_plan(path)
    for t in plan['tasks']:execute(path,t['index'])
    report=report_comparison(path,path.parent/'report',plots=False)
    assert report['completed_new_fits']==136 and not report['incomplete']
    aggregates=list(csv.DictReader((path.parent/'report/aggregate_metrics.csv').open()))
    assert all(r['completed_seeds']=='2' for r in aggregates)
    trajectories=list(csv.DictReader((path.parent/'report/trajectory_summary.csv').open()))
    assert all(r['complete_trajectory']=='True' and r['completed_cycles']=='9' for r in trajectories)
    assert all(float(r['largest_downward_step_pp'])==0 for r in trajectories)
    differences=list(csv.DictReader((path.parent/'report/paired_differences.csv').open()))
    gains=[float(r['delta_pp']) for r in differences if r['comparison']=='warm_start minus random_restart' and r['metric']=='f1_macro']
    assert gains and all(abs(g-2)<1e-10 for g in gains)
