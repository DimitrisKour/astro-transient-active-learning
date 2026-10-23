"""CPU-only contract checks for isolated representation 5B/5C package."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parents[2]

def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

manager = load('local_rep5bc', 'scripts/adaptive/rep5bc.py')
builder = load('local_rep5bc_builder', 'scripts/adaptive/rep5bc_build_runner.py')
tuning = load('local_rep5bc_vit', 'src/al_pipeline/astronomy/representation/vi_tuning.py')

ORIGINAL = '''def _freeze_expert_towers(model):
    for module_name in ("lc_tower", "spatial_tower", "coord_tower"):
        module = getattr(model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = False
    for parameter in model.vit_tower.parameters():
        parameter.requires_grad = True

def _set_training_mode(model, *, legacy_frozen_expert_train_mode):
    model.train()
    if legacy_frozen_expert_train_mode:
        return
    model.lc_tower.eval()
    model.spatial_tower.eval()
    model.coord_tower.eval()
    model.vit_tower.train()
'''

class Fake(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_tower = nn.Module()
        self.vit_tower.encoder = nn.Sequential(nn.Linear(4,4), nn.Dropout(p=0.1))
        self.vit_tower.fc_out = nn.Linear(4,5)
        self.lc_tower = nn.Linear(4,4)
        self.spatial_tower = nn.Linear(4,4)
        self.coord_tower = nn.Linear(4,4)


def test_scientific_head_only_is_actually_frozen(tmp_path):
    compiled = compile(manager.patched_training_source(ORIGINAL.encode()), "patched", "exec")
    namespace = {}
    exec(compiled, namespace)
    for mode in ("vit_head_only", "vit_full"):
        model = Fake()
        model._applecider_vit_head_only = (mode == "vit_head_only")
        namespace['_freeze_expert_towers'](model)
        namespace['_set_training_mode'](model, legacy_frozen_expert_train_mode=False)
        assert all(not p.requires_grad for p in model.lc_tower.parameters())
        if mode == "vit_head_only":
            assert all(not p.requires_grad for p in model.vit_tower.encoder.parameters())
            assert all(p.requires_grad for p in model.vit_tower.fc_out.parameters())
            assert not model.vit_tower.training
            assert model.vit_tower.fc_out.training
            assert not model.vit_tower.encoder.training
        else:
            assert all(p.requires_grad for p in model.vit_tower.parameters())
            assert model.vit_tower.training


def test_factory_sets_checkpoint_compatible_flag(monkeypatch):
    from types import ModuleType
    import sys
    modules = []
    for name in ('al_pipeline','al_pipeline.astronomy','al_pipeline.astronomy.astrominn',
                 'al_pipeline.astronomy.astrominn.inference',
                 'al_pipeline.astronomy.astrominn.inference.astrominn_vit'):
        m = ModuleType(name)
        m.__path__ = []
        monkeypatch.setitem(sys.modules, name, m)
    module = sys.modules['al_pipeline.astronomy.astrominn.inference.astrominn_vit']
    def fake_load(**kwargs):
        assert kwargs['vit_checkpoint'] == Path('l0.pt')
        return Fake()
    module.load_astrominn_vit_model = fake_load
    for mode in tuning.MODES:
        model = tuning.make_vit_factory(mode, artifact_dir='/x', legacy_code_dir='/y')(True,Path('l0.pt'))
        assert model._applecider_vit_head_only == (mode == 'vit_head_only')
        if mode == 'vit_head_only':
            assert not next(model.vit_tower.encoder.parameters()).requires_grad
    with pytest.raises(ValueError):
        tuning.make_vit_factory('full_expert_unfreeze', artifact_dir='/x', legacy_code_dir='/y')


def test_freeze_does_not_modify_upstream_and_refuses_changes(tmp_path):
    repo=tmp_path/'repo'; work=tmp_path/'work'
    src=repo/'src/al_pipeline/astronomy/astrominn/training.py'
    src.parent.mkdir(parents=True)
    src.write_text(ORIGINAL)
    for file in ('scripts/adaptive/rep5bc.py','scripts/adaptive/rep5bc_worker.sbatch',
                 'scripts/adaptive/rep5bc_build_runner.py','scripts/run_representation_5c.py'):
        p=repo/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('dummy')
    manager.freeze(repo,work)
    assert src.read_text()==ORIGINAL
    copied=work/'source/src/al_pipeline/astronomy/astrominn/training.py'
    assert "_applecider_vit_head_only" in copied.read_text()
    manager.freeze(repo,work)
    copied.write_text('tampered')
    with pytest.raises(ValueError,match='changed'):
        manager.verify_source(work)


def test_training_patch_upstream_guard():
    with pytest.raises(ValueError,match='freeze routine'):
        manager.patched_training_source(b'not a trainer')


def test_closed_loop_pilot_plan_uses_fixed_budgets(tmp_path):
    work=tmp_path/'work';work.mkdir()
    plan={'suite_id':'t5c','base_suite':manager.BASE_SUITE,'registry':'/tmp/registry',
          'initialization':'warm_start','mode':'vit_head_only','cycles':5,
          'epochs':50,'patience':10,'batch_size':512,'learning_rate':1e-4,
          'excluded_manifest':'/tmp/excluded.csv'}
    cmd=manager.command_c(plan,135,'entropy',3,work)
    assert '--run' in cmd and '--cycle' in cmd and cmd[cmd.index('--cycle')+1]=='3'
    assert cmd[cmd.index('--representation-mode')+1]=='vit_head_only'
    assert cmd[cmd.index('--training-initialization')+1]=='warm_start'
    assert '--complete-pool' not in cmd
    plan['cycles']=9
    assert '--complete-pool' in manager.command_c(plan,135,'entropy',9,work)


def test_no_jobs_called_without_submit(tmp_path,monkeypatch,capsys):
    work=tmp_path/'work';work.mkdir()
    def fake_plan(*args): return {'schema':'representation_5c_v1','seeds':[135],
                                  'policies':['entropy'],'cycles':3}
    monkeypatch.setattr(manager,'load_plan',fake_plan)
    def never(*args,**kwargs): raise AssertionError('scheduler invoked during dry-run')
    monkeypatch.setattr(manager.subprocess,'check_output',never)
    manager.submit_c(SimpleNamespace(work_root=work,submit=False,walltime='00:20:00'))
    assert 'DRY RUN' in capsys.readouterr().out


def test_duplicate_or_invalid_policies_rejected():
    with pytest.raises(ValueError): manager.checked_names(['entropy','entropy'],manager.POLICIES,'policies')
    with pytest.raises(ValueError): manager.checked_names(['BADGE'],manager.POLICIES,'policies')


def test_generated_5c_runner_guarded_and_mode_specific(tmp_path):
    repo=tmp_path/'repo'
    old=repo/'scripts/run_astrominn_historical_replay.py'
    old.parent.mkdir(parents=True)
    old.write_text('original source remains unchanged')
    stub=repo/'scripts/adaptive/task4_build_runner.py'
    stub.parent.mkdir(parents=True)
    template='''\
"""Task 4: closed-loop historical replay with warm-started AstroMiNN ViT."""
from al_pipeline.core.experiments.budget import plan_cycle_budget

def parse_args():
    parser.add_argument("--training-initialization",
                        choices=("warm_start", "restart_from_random", "restart_from_baseline"),
                        default="warm_start")

def main():
    if args.bootstrap_source_suite_id == args.suite_id:
        raise ValueError("Use a distinct suite ID for Task 4.")
    if args.training_initialization != "warm_start":
        raise ValueError("Task 4 must use warm_start; do not silently change the experiment.")
    if not args.bootstrap_source_suite_id:
        raise ValueError("Task 4 requires --bootstrap-source-suite-id.")
    if not args.require_existing_bootstrap:
        raise ValueError("Task 4 must reuse an existing, validated L0 bootstrap.")
    plan = {
        "bootstrap_source_suite_id": args.bootstrap_source_suite_id,
        "vit_tower": "L0_then_previous_loss_best_checkpoint",
    }
    if args.extend_budget:
        pass
'''
    # Embed a predictable upstream-builder stub; no changes to original runner.
    stub.write_text('def build_runner(source):\n    return '+repr(template)+'\n')
    generated=builder.generate(repo)
    assert '"representation_mode": args.representation_mode' in generated
    assert 'experiment.trainer.model_factory = make_vit_factory' in generated
    assert 'restart_from_baseline' in generated
    assert 'Task 5C' in generated
    assert old.read_text() == 'original source remains unchanged'
    with pytest.raises(ValueError):
        builder.replace_once('a a','a','x')
