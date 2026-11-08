"""Task7A scientific acquisition, deterministic fits, scheduler & source isolation."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/research/historical_replay/task7a'))
import task7a as jobs
import task7a_build as build
import task7a_prepare as prep
import task7a_temperature_science as science


def test_15_independent_fit_dag():
    rows=jobs.plan()
    assert len(rows)==15 and len({r['key'] for r in rows})==15
    assert {r['policy'] for r in rows}=={'entropy'}
    for seed in (135,246,357):
        sub=[r for r in rows if r['seed']==seed]
        assert [r['cycle'] for r in sub]==[1,2,3,4,5]
        for r in sub:
            assert r['parent']==(None if r['cycle']==1 else f'{seed}:entropy:{r["cycle"]-1}')


def test_sbatch_and_walltime():
    for row in jobs.plan():
        cmd=jobs.sbatch_command(Path('/tmp/t7'),row,None if row['cycle']==1 else '123', '00:20:00')
        assert '--hold' in cmd and '--time=00:20:00' in cmd
        assert '--gres=gpu:1' not in cmd # SBATCH header, not CLI
        assert cmd[-2:]==[str(row['seed']),str(row['cycle'])]
        assert ('--dependency=afterok:123' in cmd)==(row['cycle']>1)
    for value in ('00:15:00','00:20:00'):jobs.validate_walltime(value)
    for value in ('00:00:00','00:25:00','garbage'):
        with pytest.raises(ValueError):jobs.validate_walltime(value)


def test_dry_plan_never_submits(monkeypatch):
    monkeypatch.setattr(jobs,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Called external tool')))
    assert len(jobs.plan())==15


def test_fit_temperature_smaller_when_logits_underconfident():
    y=np.arange(5).repeat(10)
    logits=np.zeros((50,5),dtype=np.float64)
    logits[np.arange(50),y]=0.4
    temp=science.fit_temperature(logits,y)
    assert 0.05<=temp<1
    assert science.nll(logits,y,temp)<science.nll(logits,y,1)
    assert abs(science.fit_temperature(logits,y)-temp)<1e-9


def test_calibration_uses_object_logits_not_sample_rows(tmp_path):
    ck=tmp_path/'best_vit_tower.pth';ck.write_bytes(b'checkpoint')
    ev=tmp_path/'evaluation';ev.mkdir()
    n=30;y=np.tile(np.arange(5),6)
    z=np.full((n,5),-1.);z[np.arange(n),y]=1.2
    # Irrelevant extra per-sample fields have DIFFERENT logits and counts.
    np.savez_compressed(ev/'validation_predictions.npz',object_ids=np.array([f'o{i}' for i in range(n)]),
        object_targets=y,object_values=z,sample_targets=np.zeros(200,dtype=int),
        sample_values=np.ones((200,5)))
    summary={'checkpoint_path':str(ck),'checkpoint_sha256':science.digest(ck),
             'model_class_names':list(science.CLASSES),'splits':[{'split':'validation'}]}
    (ev/'evaluation_summary.json').write_text(json.dumps(summary))
    out=tmp_path/'temp.json'
    t=science.fitted_temperature(cycle=1,checkpoint=ck,evaluation_dir=ev,record_path=out)
    assert abs(t-science.fit_temperature(z,y))<1e-9
    assert json.loads(out.read_text())['calibration']['n_validation_objects']==30
    assert science.fitted_temperature(cycle=1,checkpoint=ck,evaluation_dir=ev,record_path=out)==t
    ck.write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest'):
        science.fitted_temperature(cycle=1,checkpoint=ck,evaluation_dir=ev,record_path=out)


def test_corrupted_eval_predictions_rejected(tmp_path):
    ck=tmp_path/'ck';ck.write_bytes(b'a')
    ev=tmp_path/'e';ev.mkdir()
    (ev/'evaluation_summary.json').write_text(json.dumps({'checkpoint_path':str(ck),
        'checkpoint_sha256':science.digest(ck),'model_class_names':list(science.CLASSES),
        'splits':[{'split':'validation'}]}))
    y=np.tile(np.arange(5),6)
    np.savez(ev/'validation_predictions.npz',object_ids=np.array(['dup']*30),object_targets=y,object_values=np.zeros((30,5)))
    with pytest.raises(ValueError,match='latest-alert'):
        science.fitted_temperature(cycle=1,checkpoint=ck,evaluation_dir=ev,record_path=tmp_path/'r.json')


def test_simulated_patch_is_scientifically_isolated():
    fixture='''        self.identity_path = self.run_dir / "experiment.json"
        return inference.predictions
        payload = {
            "schema_version": 1,
'''
    out=build.simulated(fixture)
    assert 'temperature_scaling_enabled = False' in out
    assert 'inference.predictions.values / temperature' in out
    assert 'validation_object_logits_latest_alert' not in out # helper owns calibration
    assert 'task7a_temperature_scaling' in out
    assert 'self._inference_checkpoint(cycle)' in out
    assert 'temperature_calibration.json' in out


def test_common_contract_and_gpu_worker():
    common=(ROOT/'scripts/research/historical_replay/task7a/task7a_common.sh').read_text()
    worker=(ROOT/'scripts/research/historical_replay/task7a/task7a_worker.sbatch').read_text()
    assert '--bootstrap-source-suite-id "$TASK7A_BASE" --require-existing-bootstrap' in common
    assert '--training-initialization warm_start --strategies entropy' in common
    assert '--max-cycles 9 --complete-pool' in common
    assert '--cycle "$cycle" --run' in worker
    assert '--gres=gpu:1' in worker and '--time=00:20:00' in worker
    assert 'task7a_activate' in worker
    assert '--bootstrap-only' not in worker
    assert prep.SUITE != prep.REFERENCE


def test_source_preflight_order_no_run(tmp_path,monkeypatch):
    source=tmp_path/'source';(source/'scripts').mkdir(parents=True)
    (source/'scripts/run_task7a_closed_loop.py').write_text('')
    monkeypatch.setattr(jobs,'verify_frozen',lambda root:None)
    monkeypatch.setattr(jobs,'command_args',lambda root,seed:['--seed',str(seed)])
    monkeypatch.setattr(jobs,'scientific_env',lambda root:{})
    seen=[]
    def fake_run(argv,**kwargs):
        assert '--device' in argv and 'cpu' in argv and '--run' not in argv
        seen.append('materialized')
    monkeypatch.setattr(jobs,'run',fake_run)
    monkeypatch.setattr(jobs,'check_splits',lambda p,d:seen.append('checked'))
    jobs.preflight(tmp_path)
    assert seen==['materialized']*3+['checked']


def test_runner_has_correct_upstream_anchors():
    s=(ROOT/'scripts/research/historical_replay/task7a/task7a_build.py').read_text()
    assert 'task4_build_runner.py' in s
    assert 'if args.bootstrap_source_suite_id !=' in s
    assert 'class_weight_exponent=0.85' in s
    assert 'experiment.temperature_scaling_enabled = True' in s
    assert 'cycle_training_initialization' in s


def test_immutable_freeze(tmp_path,monkeypatch):
    # Freeze must not modify the editable repo, while refusing tampered snapshots.
    repo=tmp_path/'repo';src=repo/'src/al_pipeline/astronomy/experiments';src.mkdir(parents=True)
    (src/'simulated.py').write_text('original source')
    scripts=repo/'scripts/research/historical_replay/task7a';scripts.mkdir(parents=True)
    for name in ('task7a.py','task7a_build.py','task7a_prepare.py','task7a_common.sh','task7a_worker.sbatch','task7a_temperature_science.py'):
        (scripts/name).write_text('x')
    monkeypatch.setattr(prep,'build_snapshot_files',lambda r:('runner','modified simulation','temperature science'))
    monkeypatch.setattr(Path,'is_relative_to',lambda self,other:True)
    reg=tmp_path/'registry.toml';reg.write_text('valid registry')
    job=tmp_path/'output'
    prep.freeze(repo,job,reg)
    assert (src/'simulated.py').read_text()=='original source'
    assert (job/'source/src/al_pipeline/astronomy/experiments/simulated.py').read_text()=='modified simulation'
    assert (job/'source/src/al_pipeline/astronomy/experiments/temperature_acquisition.py').read_text()=='temperature science'
    assert (job/'source/scripts/research/historical_replay/task7a/task7a.py').is_file()
    prep.freeze(repo,job,reg)
    (job/'source/src/al_pipeline/astronomy/experiments/simulated.py').write_text('tampered')
    with pytest.raises(ValueError,match='damaged'):
        prep.verify(job)
