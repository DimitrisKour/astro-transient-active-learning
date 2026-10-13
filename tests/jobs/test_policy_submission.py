"""Exercise the actual submission shell without accessing Slurm."""

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
    persistent = tmp_path / "persistent"
    for name in ("policy_common.sh", "submit_policy_compare.sh", "policy_cycle.sbatch"):
        content = (
            (ROOT / "scripts/adaptive" / name)
            .read_text()
            .replace("/projects/bcrv/dkourtesis/applecider_adaptive", str(persistent))
        )
        (scripts / name).write_text(content)
    job_dir = persistent / "cycle_jobs/astrominn_policy_compare_v1"
    (job_dir / "logs").mkdir(parents=True)
    (job_dir / "registry_hdd.toml").write_text(f'[storage]\npersistent_root="{persistent}"\n')
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("module", "quota", "squeue"):
        executable(commands / name, "#!/bin/bash\nexit 0\n")
    executable(commands / "conda", "#!/bin/bash\nexit 0\n")
    executable(
        commands / "python",
        f"""#!/bin/bash
case "$1" in
  scripts/adaptive/prepare_policy_compare.py) exit 0;;
  scripts/run_astrominn_historical_replay.py) echo '{{"max_cycles":9}}'; exit 0;;
  *) exec {shutil.which(sys.executable)} "$@";;
esac
""",
    )
    executable(
        commands / "sbatch",
        f"""#!{sys.executable}
import json,os,sys
from pathlib import Path
p=Path(os.environ['MOCK_COUNTER'])
n=int(p.read_text())+1 if p.exists() else 1
p.write_text(str(n))
with open(os.environ['MOCK_LOG'],'a') as out:
    out.write(json.dumps(['sbatch',*sys.argv[1:]])+'\\n')
if n == int(os.environ.get('MOCK_FAIL_AT','0')): sys.exit(1)
print(1000+n)
""",
    )
    for name in ("scontrol", "scancel"):
        executable(
            commands / name,
            f"""#!{sys.executable}
import json,os,sys
with open(os.environ['MOCK_LOG'],'a') as out:
    out.write(json.dumps(['{name}',*sys.argv[1:]])+'\\n')
""",
        )
    # Isolate mocked commands from cluster shell initialization.
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key not in {"BASH_ENV", "ENV"} and not key.startswith("BASH_FUNC_")
    }
    env = dict(
        inherited,
        POLICY_SUITE_ID="astrominn_policy_compare_v1",
        PATH=f"{commands}:{os.environ['PATH']}",
        CONDA_PREFIX=str(tmp_path / "conda"),
        MOCK_COUNTER=str(tmp_path / "counter"),
        MOCK_LOG=str(tmp_path / "commands.log"),
    )
    return scripts, env, job_dir


def test_submission_holds_nine_jobs_and_releases_in_reverse_order(tmp_path):
    scripts, env, job_dir = setup(tmp_path)
    result = subprocess.run(
        ["bash", str(scripts / "submit_policy_compare.sh"), "246", "00:35:00"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["MOCK_LOG"]).read_text().splitlines()]
    submissions = [c for c in calls if c[0] == "sbatch"]
    assert len(submissions) == 9
    assert all("--hold" in c and "--time=00:35:00" in c for c in submissions)
    for i, call in enumerate(submissions[1:], start=1):
        assert f"--dependency=afterok:{1000 + i}" in call
    releases = [c for c in calls if c[0] == "scontrol"]
    assert [c[-1] for c in releases] == [str(n) for n in range(1009, 1000, -1)]
    assert not any(c[0] == "scancel" for c in calls)
    duplicate = subprocess.run(
        ["bash", str(scripts / "submit_policy_compare.sh"), "246"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert duplicate.returncode != 0
    assert "already exists" in duplicate.stderr


def test_submission_failure_cancels_only_new_held_jobs(tmp_path):
    scripts, env, _ = setup(tmp_path)
    env["MOCK_FAIL_AT"] = "3"
    result = subprocess.run(
        ["bash", str(scripts / "submit_policy_compare.sh"), "357"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    calls = [json.loads(line) for line in Path(env["MOCK_LOG"]).read_text().splitlines()]
    assert ["scancel", "1001", "1002"] in calls
    assert not any(c[0] == "scontrol" for c in calls)


def test_prepare_only_does_not_submit(tmp_path):
    scripts, env, _ = setup(tmp_path)
    result = subprocess.run(
        ["bash", str(scripts / "submit_policy_compare.sh"), "135", "--prepare-only"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["MOCK_LOG"]).exists()


def test_submission_ignores_inherited_shell_initialization(tmp_path, monkeypatch):
    startup = tmp_path / "cluster_startup.sh"
    startup.write_text("python() { return 91; }; module() { return 92; }\n")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("ENV", str(startup))
    monkeypatch.setenv("BASH_FUNC_module%%", "() { return 93; }")
    monkeypatch.setenv("BASH_FUNC_conda%%", "() { return 94; }")
    scripts, env, _ = setup(tmp_path)
    assert "BASH_ENV" not in env and "ENV" not in env
    assert not any(key.startswith("BASH_FUNC_") for key in env)
    result = subprocess.run(
        ["bash", str(scripts / "submit_policy_compare.sh"), "135", "--prepare-only"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["MOCK_LOG"]).exists()


def test_new_policy_chain_does_not_reject_or_submit_completed_baselines(tmp_path):
    scripts, env, original_job_dir = setup(tmp_path)
    suite = "astrominn_historical_replay_v1"
    env["POLICY_SUITE_ID"] = suite
    persistent = original_job_dir.parents[1]
    job_dir = original_job_dir.parent / suite
    (job_dir / "logs").mkdir(parents=True)
    (job_dir / "registry_hdd.toml").write_text(
        f'[storage]\npersistent_root="{persistent}"\n'
    )
    for strategy in ("random", "entropy"):
        p = persistent / "runs" / suite / strategy / "seed_135"
        p.mkdir(parents=True)
        (p / "run_state.json").write_text(
            json.dumps(dict(last_completed_cycle=8, active_cycle=None, status="ready"))
        )
    result = subprocess.run(["bash", str(scripts / "submit_policy_compare.sh"), "135"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    arguments = subprocess.run(
        ["bash", "-c", 'source "$1"; policy_arguments 135; printf "%s\\n" "${POLICY_ARGS[@]}"',
         "bash", str(scripts / "policy_common.sh")],
        env=env, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    start = arguments.index("--strategies") + 1
    stop = next(i for i in range(start, len(arguments)) if arguments[i].startswith("--"))
    assert arguments[start:stop] == ["entropy_diversity", "entropy_balanced_explore"]
    assert "--require-existing-bootstrap" in arguments
