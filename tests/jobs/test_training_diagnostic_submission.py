"""Exercise the real launcher with fake Slurm; no remote jobs are submitted."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def setup(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts/adaptive"
    scripts.mkdir(parents=True)
    launcher = scripts / "submit_training_diagnostics.sh"
    shutil.copyfile(ROOT / "scripts/adaptive/submit_training_diagnostics.sh", launcher)
    commands = tmp_path / "bin"
    commands.mkdir()
    prepare = tmp_path / "fake_prepare.py"
    prepare.write_text('''
import hashlib,json,sys
from pathlib import Path
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
root=Path(args['--persistent-root'])/'training_diagnostics'/args['--diagnostic-id']
root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
path=root/'plan.json'
if not path.exists():
    tasks=[]
    for index in range(57):
        tasks.append({'index':index,'task_id':f'fit_{index:03d}',
                      'kind':'transition' if index<42 else 'order','anchor':index<2})
    payload={'schema_version':1,'metadata_root':str(root),'tasks':tasks}
    path.write_text(json.dumps(payload,sort_keys=True))
    (root/'plan.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    frozen=root/'source/scripts/adaptive';frozen.mkdir(parents=True)
    (frozen/'training_diagnostic.sbatch').write_text('#!/bin/bash\\n')
print(path)
''')
    executable(commands / "python", f'''#!/bin/bash
if [[ "$1" == *prepare_training_diagnostics.py ]]; then
  shift
  exec {sys.executable} {prepare} "$@"
fi
exec {sys.executable} "$@"
''')
    executable(commands / "sbatch", f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
p=Path(os.environ['MOCK_COUNTER']);n=int(p.read_text())+1 if p.exists() else 1
p.write_text(str(n))
with open(os.environ['MOCK_LOG'],'a') as out:
    out.write(json.dumps(['sbatch',*sys.argv[1:]])+'\\n')
if n==int(os.environ.get('MOCK_FAIL_AT','0')):sys.exit(1)
print(1000+n)
''')
    for command in ("scontrol", "scancel"):
        executable(commands / command, f'''#!{sys.executable}
import json,os,sys
with open(os.environ['MOCK_LOG'],'a') as out:
    out.write(json.dumps(['{command}',*sys.argv[1:]])+'\\n')
if '{command}'=='scontrol' and sys.argv[-1]==os.environ.get('MOCK_RELEASE_FAIL',''):sys.exit(1)
''')
    executable(commands / "squeue", '''#!/bin/bash
if [[ "$*" == *--noheader* ]]; then
  [[ -z "${MOCK_ACTIVE:-}" ]] || echo "$MOCK_ACTIVE"
fi
exit 0
''')
    inherited = {k: v for k, v in os.environ.items()
                 if k not in {"BASH_ENV", "ENV"} and not k.startswith("BASH_FUNC_")}
    env = dict(inherited, DIAG_SKIP_ACTIVATE="1", DIAG_ID="test_diag",
               DIAG_REGISTRY=str(tmp_path / "registry.toml"),
               DIAG_PERSISTENT_ROOT=str(tmp_path / "persistent"),
               DIAG_DATA_ROOT=str(tmp_path / "heavy"),
               PATH=f"{commands}:{os.environ['PATH']}",
               PYTHONPATH=f"{ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
               MOCK_COUNTER=str(tmp_path / "counter"), MOCK_LOG=str(tmp_path / "calls.jsonl"))
    return launcher, env, tmp_path / "persistent/training_diagnostics/test_diag"


def invoke(launcher, env, *args):
    return subprocess.run(["bash", str(launcher), *args], env=env, text=True, capture_output=True)


def calls(env):
    path = Path(env["MOCK_LOG"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_submit_57_fits_in_held_arrays_with_anchor_dependency(tmp_path):
    launcher, env, metadata = setup(tmp_path)
    result = invoke(launcher, env)
    assert result.returncode == 0, result.stderr
    events = calls(env)
    submissions = [r for r in events if r[0] == "sbatch"]
    assert len(submissions) == 3
    assert all("--hold" in row for row in submissions)
    assert "--time=00:15:00" in submissions[0]
    assert "--time=00:15:00" in submissions[1]
    assert "--time=00:20:00" in submissions[2]
    assert "--array=0,1%2" in submissions[0]
    assert "--array=" + ",".join(map(str, range(2, 42))) + "%3" in submissions[1]
    assert "--array=" + ",".join(map(str, range(42, 57))) + "%1" in submissions[2]
    assert not any(r.startswith("--dependency") for r in submissions[0])
    assert all("--dependency=afterok:1001" in r for r in submissions[1:])
    assert [r for r in events if r[0] == "scontrol"] == [
        ["scontrol", "release", "1003"], ["scontrol", "release", "1002"],
        ["scontrol", "release", "1001"],
    ]
    assert all(str(metadata / "source") in r[-1] for r in submissions)


def test_prepare_only_has_no_scheduler_actions(tmp_path):
    launcher, env, metadata = setup(tmp_path)
    result = invoke(launcher, env, "--prepare-only")
    assert result.returncode == 0, result.stderr
    assert calls(env) == []
    assert (metadata / "plan.json").exists()
    assert not (metadata / "submissions.tsv").exists()


def test_duplicate_submission_refused(tmp_path):
    launcher, env, metadata = setup(tmp_path)
    assert invoke(launcher, env).returncode == 0
    before = calls(env)
    result = invoke(launcher, env)
    assert result.returncode != 0
    assert "Submission record exists" in result.stderr
    assert calls(env) == before


def test_submission_failure_cancels_only_this_attempt(tmp_path):
    launcher, env, metadata = setup(tmp_path)
    env["MOCK_FAIL_AT"] = "2"
    assert invoke(launcher, env).returncode != 0
    events = calls(env)
    assert [r for r in events if r[0] == "scancel"] == [["scancel", "1001"]]
    assert not [r for r in events if r[0] == "scontrol"]
    assert (metadata / "submissions.tsv").exists()


def test_release_failure_cancels_all_new_arrays(tmp_path):
    launcher, env, metadata = setup(tmp_path)
    env["MOCK_RELEASE_FAIL"] = "1002"
    assert invoke(launcher, env).returncode != 0
    assert [r for r in calls(env) if r[0] == "scancel"] == [
        ["scancel", "1001", "1002", "1003"],
    ]


def test_retry_refuses_active_jobs_then_queues_only_missing_fits(tmp_path):
    launcher, env, metadata = setup(tmp_path)
    assert invoke(launcher, env).returncode == 0
    env["MOCK_ACTIVE"] = "1002_5"
    before = calls(env)
    assert invoke(launcher, env, "--retry-incomplete").returncode != 0
    assert calls(env) == before
    env.pop("MOCK_ACTIVE")
    plan = json.loads((metadata / "plan.json").read_text())
    digest = hashlib.sha256((metadata / "plan.json").read_bytes()).hexdigest()
    for task in plan["tasks"]:
        if task["index"] in (5, 55):
            continue
        directory = metadata / "tasks" / task["task_id"]
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({"task": task, "plan_sha256": digest}))
    result = invoke(launcher, env, "00:25:00", "00:30:00", "--retry-incomplete")
    assert result.returncode == 0, result.stderr
    remaining = [r for r in calls(env)[len(before):] if r[0] == "sbatch"]
    assert len(remaining) == 2
    assert "--array=5%3" in remaining[0]
    assert "--time=00:25:00" in remaining[0]
    assert "--array=55%1" in remaining[1]
    assert not any(arg.startswith("--dependency") for row in remaining for arg in row)


def test_invalid_walltime_rejected_before_submission(tmp_path):
    launcher, env, metadata = setup(tmp_path)
    result = invoke(launcher, env, "7 minutes")
    assert result.returncode != 0
    assert calls(env) == []

