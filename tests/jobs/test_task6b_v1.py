"""Task 6B DAG, scientific-source patch and object macro-F1 smoke tests."""
from __future__ import annotations
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'scripts/research/historical_replay/task6b'
sys.path.insert(0,str(SCRIPTS))
import task6b as jobs
import task6b_build as build
import task6b_prepare as prepare


def test_57_jobs_and_correct_dependencies():
    rows=jobs.plan()
    assert len(rows)==57
    lookup={x['key']:x for x in rows}
    assert len(lookup)==57
    assert sum(x['cycle']==0 for x in rows)==3
    assert sum(x['policy']=='entropy' for x in rows)==27
    assert sum(x['policy']=='entropy_balanced_explore' for x in rows)==27
    for seed in (135,246,357):
        assert lookup[f'l0:{seed}']['parent'] is None
        for pol in jobs.POLICIES:
            assert lookup[f'{seed}:{pol}:1']['parent']==f'l0:{seed}'
            for cy in range(2,10):
                assert lookup[f'{seed}:{pol}:{cy}']['parent']==f'{seed}:{pol}:{cy-1}'


def test_sbatch_isolated_one_fit_per_job_and_short_walltime():
    for x in jobs.plan():
        dep=None if x['cycle']==0 else '500'
        args=jobs.sbatch_command(Path('/tmp/6b'),Path('/tmp/6b/source'),x,
                                 parent_job=dep,walltime='00:20:00',l0_walltime='00:20:00')
        assert '--hold' in args and '--parsable' in args
        assert '--time=00:20:00' in args
        if dep: assert args.count('--dependency=afterok:500')==1
        else: assert not any(y.startswith('--dependency=') for y in args)
        assert args[-3:]==[str(x['seed']),str(x['cycle']),x['policy']]
        assert 'task6b_worker.sbatch' in args[-5]
    for t in ('00:15:00','00:20:00'): jobs.validate_walltime(t)
    for t in ('00:00:00','00:25:00','03:00:00','nonsense'):
        with pytest.raises(ValueError): jobs.validate_walltime(t)


def test_readonly_plan_does_not_contact_scheduler(monkeypatch,capsys):
    monkeypatch.setattr(jobs,'_run',lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('scheduler called')))
    # plan() itself is read-only and does not inspect any job directory
    assert len(jobs.plan(through=2))==15


def test_science_contract_and_warmstart():
    assert prepare.SUITE=='astrominn_closed_loop_warm_f1best_085_v1'
    assert prepare.REFERENCE=='astrominn_closed_loop_warm_v1'
    assert jobs.POLICIES==('entropy','entropy_balanced_explore')
    common=(SCRIPTS/'task6b_common.sh').read_text()
    worker=(SCRIPTS/'task6b_worker.sbatch').read_text()
    assert '--bootstrap-source-suite-id "$TASK6B_SUITE_ID"' in common
    assert '--balance-beta 0.5' in common
    assert '--training-initialization warm_start' in common
    assert '--bootstrap-only --run' in worker
    assert '--require-existing-bootstrap --cycle "$cycle" --run' in worker
    assert 'best_object_macro_f1' in (SCRIPTS/'task6b.py').read_text()
    assert 'best_loss_vit_tower.pth' in (SCRIPTS/'task6b_report.py').read_text()


TRAINING_FIXTURE = '''from sklearn.metrics import auc, precision_recall_curve
    class_weight_source: ClassWeightSource = "legacy_reference"
            raise ValueError(
                "class_weight_source must be 'legacy_reference' or 'training_manifest'."
            )
    @classmethod
    def legacy_seed135(
        checkpoint_path = request.output_dir / "best_vit_tower.pth"
        best_validation_loss = float("inf")
        best_epoch = 0
            validation_loss = _evaluate_loss(
                loader=validation_loader,
                model=model,
                criterion=criterion,
                device=self.device,
            )
            improved = validation_loss < best_validation_loss

            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                _torch_save_atomic(
                    model.vit_tower.state_dict(),
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                "validation_loss": validation_loss,
                    "best_epoch": best_epoch,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
        result = TrainingResult(
            checkpoint_path=Path(summary["checkpoint_path"]),
def _evaluate_auprc(
'''


def test_training_patch_contains_loss_and_f1_selection():
    patched=build.patch_training(TRAINING_FIXTURE)
    assert 'checkpoint_selection: Literal["loss_best", "object_macro_f1_best"]' in patched
    assert 'best_loss_vit_tower.pth' in patched
    assert 'scheduler.step(validation_loss)' not in TRAINING_FIXTURE  # unchanged in actual source
    assert 'f1_score(' in patched
    assert 'checkpoint_selection' in patched
    assert 'best_loss_epoch' in patched
    assert 'return total_loss / len(loader), score' in patched
    assert '_task6b_evaluate_loss_and_object_f1' in patched
    assert 'latest[key] = record' in patched


def test_latest_alert_object_f1_is_not_sample_f1():
    torch=pytest.importorskip('torch')
    sk=pytest.importorskip('sklearn.metrics')
    patched=build.patch_training(TRAINING_FIXTURE)
    start=patched.index('def _task6b_evaluate_loss_and_object_f1(')
    end=patched.index('\ndef _evaluate_auprc(',start)
    snippet='from __future__ import annotations\n'+patched[start:end]
    env={'torch':torch, 'nn':torch.nn, 'DataLoader':object,
         'f1_score':sk.f1_score, 'MODEL_CLASS_NAMES':('AGN','TDE','SN I','SN II','CV'),
         '_extract_logits':lambda x:x['logits']}
    exec(compile(snippet,'generated_helper','exec'),env)
    f=env['_task6b_evaluate_loss_and_object_f1']
    class Dummy:
        def eval(self):pass
        def __call__(self, metadata, *,image):
            return {'logits':metadata}
    logits=torch.tensor([[8.,0,0,0,0],[0,8.,0,0,0],[0,8.,0,0,0],
                         [0,0,8.,0,0],[0,0,0,8.,0],[0,0,0,0,8.]],dtype=torch.float32)
    ids=['A','A','B','C','D','E']
    targets=torch.tensor([0,0,1,2,3,4])
    batch={'metadata':logits,'image':torch.zeros(6,1),'target':torch.nn.functional.one_hot(targets,num_classes=5).float(),
           'target_index':targets,'obj_id':ids,'alert':torch.tensor([1,2,1,1,1,1])}
    loss, actual=f(loader=[batch],model=Dummy(),criterion=torch.nn.CrossEntropyLoss(reduction='none'),device=torch.device('cpu'))
    # Latest A has class-1 prediction although its earliest alert was correct.
    expected=sk.f1_score([0,1,2,3,4],[1,1,2,3,4],labels=list(range(5)),average='macro',zero_division=0)
    assert np.isclose(actual,expected)
    assert loss>0
    # Deliberately different from sample-level F1, which includes A's correct first alert.
    sample=sk.f1_score(targets.numpy(),logits.argmax(axis=1).numpy(),labels=list(range(5)),average='macro',zero_division=0)
    assert actual!=sample


def test_report_checks_selection_and_provenance():
    s=(SCRIPTS/'task6b_report.py').read_text()
    assert "get('checkpoint_selection')!='object_macro_f1_best'" in s
    assert "warm-start" in s.lower()
    assert "L0 differs" in s


def test_freeze_is_isolated_and_immutable(tmp_path,monkeypatch):
    repo=tmp_path/'repo'
    package=repo/'src/al_pipeline/astronomy/astrominn'
    package.mkdir(parents=True)
    (package/'training.py').write_text('upstream training')
    scripts=repo/'scripts/research/historical_replay/task6b'
    scripts.mkdir(parents=True)
    for name in ('task6b.py','task6b_build.py','task6b_prepare.py','task6b_common.sh','task6b_worker.sbatch'):
        (scripts/name).write_text(name)
    monkeypatch.setattr(prepare,'build_snapshot_files', lambda r: ('patched runner', 'patched training'))
    registry=tmp_path/'registry.toml'
    registry.write_text('[storage]\npersistent_root="p"\ndata_root="d"\n')
    target=tmp_path/'j'
    prepare.freeze(repo,target,registry)
    assert (target/'source/scripts/run_task6b_closed_loop.py').read_text()=='patched runner'
    assert (target/'source/src/al_pipeline/astronomy/astrominn/training.py').read_text()=='patched training'
    assert (repo/'src/al_pipeline/astronomy/astrominn/training.py').read_text()=='upstream training'
    prepare.verify(target)
    prepare.freeze(repo,target,registry)
    (target/'source/src/al_pipeline/astronomy/astrominn/training.py').write_text('tampered')
    with pytest.raises(ValueError,match='mismatch'):
        prepare.verify(target)


def test_preflight_materializes_before_split_comparison(tmp_path,monkeypatch):
    source=tmp_path/'source'
    (source/'scripts/research/historical_replay/task6b').mkdir(parents=True)
    (source/'scripts/run_task6b_closed_loop.py').write_text('x')
    (source/'src').mkdir(parents=True)
    (source/'scripts/research/historical_replay/task6b/task6b_common.sh').write_text('')
    monkeypatch.setattr(jobs,'_arguments',lambda root,seed,policy:['--fake',str(seed)])
    monkeypatch.setattr(jobs,'scientific_env',lambda root:{})
    seen=[]
    def fake_run(args,**kwargs):
        seen.append('materialize')
        assert '--device' in args and 'cpu' in args
        assert '--run' not in args
    monkeypatch.setattr(jobs,'_run',fake_run)
    monkeypatch.setattr(jobs,'check_splits',lambda p,d: seen.append('check_splits'))
    jobs.source_preflight(tmp_path,tmp_path)
    assert seen==['materialize']*3+['check_splits']
