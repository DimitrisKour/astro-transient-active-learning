"""Storage routing preserves inputs and rejects changes to frozen suites."""

from pathlib import Path
import subprocess
import sys
import tomllib
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("reuse_registry", [False, True])
def test_preflight_preserves_dataset_paths_and_detects_source_changes(tmp_path, reuse_registry):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "src/al_pipeline/core").mkdir(parents=True)
    (repo / "scripts/adaptive").mkdir(parents=True)
    core = repo / "src/al_pipeline/core/example.py"
    core.write_text("value = 1\n")
    (repo / "scripts/run_astrominn_historical_replay.py").write_text("# launcher\n")
    persistent, data = tmp_path / "persistent", tmp_path / "hdd"
    content = (ROOT / "scripts/adaptive/prepare_policy_compare.py").read_text()
    content = content.replace(
        "/projects/bcrv/dkourtesis/applecider_adaptive", str(persistent)
    ).replace("/work/hdd/bcrv/dkourtesis/applecider_adaptive", str(data))
    preflight = repo / "scripts/adaptive/prepare_policy_compare.py"
    preflight.write_text(content)
    (repo / "configs/delta.toml").write_text("""[storage]
persistent_root = "/old/state"
data_root = "/old/nvme/output"
[datasets.toy]
data_root = "/work/nvme/immutable/inputs"
train_manifest = "/immutable/train.csv"
""")
    job_dir = persistent / "jobs"
    args = [sys.executable, str(preflight), "--repo", str(repo), "--job-dir", str(job_dir)]
    if reuse_registry:
        source = tmp_path / "original_replay_registry.toml"
        (repo / "configs/delta.toml").rename(source)
        args += ["--source-registry", str(source)]
    first = subprocess.run(args, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    routed = tomllib.loads((job_dir / "registry_hdd.toml").read_text())
    assert routed["datasets"]["toy"]["data_root"] == "/work/nvme/immutable/inputs"
    assert routed["storage"]["data_root"] == str(data)
    second = subprocess.run(args + ["--verify-source"], text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    core.write_text("value = 2\n")
    changed = subprocess.run(args + ["--verify-source"], text=True, capture_output=True)
    assert changed.returncode != 0
    assert "Code changed" in changed.stderr
