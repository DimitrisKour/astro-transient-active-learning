"""Task 0/1 migration contracts: protect frozen live Slurm paths and isolate new suites.

This suite must run without Delta datasets, model weights, broker credentials,
Slurm, or GPU submissions. The original scientific contracts are additionally
tested by test_task0_full_history_v1.py and test_task1_closed_loop_v1.py.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "scripts/adaptive"
NEW = ROOT / "scripts/research/fusion_image"
LEGACY_GIT_BLOBS = {
    "submit_task0_final_test.sh": "8f09a24b703f6d6613dce94782072a8bc40e43eb",
    "submit_task0_full_history.sh": "ec07a9202058759909ae6a3b150bbf00b6c5380b",
    "submit_task0_test_cache.sh": "0eb9ec3486a628045c89a96bcfbc4b0420f4fcdf",
    "submit_task1_closed_loop.sh": "0710db78da5df1d92fe68c5c70cc5836a8c09df3",
    "task0_final_test_eval.sbatch": "ddc498f09574e277da5581fba9febf3eda82deb7",
    "task0_full_history.py": "4dd620511c085b7b291964e286e502be1eeed225",
    "task0_full_history.sbatch": "1d8621cd22eccb2f6d56bf6a6ba51aadb3b8da86",
    "task0_test_cache_consolidate.sbatch": "91c48a27b10222127a37c169ad58a8218b465009",
    "task0_test_cache_shard.sbatch": "10896b91a2b26b9b1c1643e56753cedc0478c4e1",
    "task0_test_cache_v1.py": "303ab0c00c6cc0436b3f256cd964d19fa8498925",
    "task0_vit_test_compare.py": "a346018d30b20c70ea244d0d4e06584355dac3f6",
    "task1_closed_loop.py": "e04f9d127fe77123dcacef909394285a302dfef6",
    "task1_cycle.sbatch": "48b0e473a8b983dfd56ad1cd4f67f4d5813424ba",
    "task1_schedule.py": "0f1ccfccefe6dee67c9741e4798e7f6f2fa8d9e6",
}


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\\0" + data).hexdigest()


@pytest.mark.parametrize("name,sha", LEGACY_GIT_BLOBS.items())
def test_legacy_job_paths_are_byte_for_byte_unchanged(name: str, sha: str):
    assert git_blob_sha(OLD / name) == sha


def test_grouped_files_exist_and_fingerprint_is_separate():
    task0 = [name for name in LEGACY_GIT_BLOBS if "task0" in name]
    task1 = [name for name in LEGACY_GIT_BLOBS if "task1" in name]
    assert len(task0) == 10
    assert len(task1) == 4
    for task, names in (("task0", task0), ("task1", task1)):
        for name in names:
            path = NEW / task / name
            assert path.is_file(), path
            assert git_blob_sha(path) != LEGACY_GIT_BLOBS[name]


def test_grouped_paths_are_self_consistent_and_not_old_write_roots():
    for task, names in (
        ("task0", [n for n in LEGACY_GIT_BLOBS if "task0" in n]),
        ("task1", [n for n in LEGACY_GIT_BLOBS if "task1" in n]),
    ):
        base = NEW / task
        for name in names:
            text = (base / name).read_text()
            for n in names:
                assert f"scripts/adaptive/{n}" not in text, (name, n)
        assert "scripts/adaptive/" not in (base / f"{task}_closed_loop.py").read_text() if task == "task1" else True

    t0 = (NEW / "task0" / "task0_full_history.py").read_text()
    assert "new_image_task0_full_historical_grouped_v1" in t0
    assert "new_image_task0_full_historical_v1" not in t0
    cache = (NEW / "task0" / "task0_test_cache_v1.py").read_text()
    assert "GROUPED_TEST_CACHE" in cache
    assert "new_image_historical_test_fastcache_grouped_v1" in cache
    al = (NEW / "task1" / "task1_closed_loop.py").read_text()
    assert "'scripts/research/fusion_image/task1/task1_closed_loop.py'" in al
    assert "new_image_task1_closed_loop_grouped_v1" in al
    assert "new_image_task0_full_historical_grouped_v1" in al
    assert "new_image_historical_test_fastcache_grouped_v1" in al
    assert "new_image_task1_closed_loop_v1" not in al


def test_new_sbatch_scripts_run_only_grouped_python():
    for task in ("task0", "task1"):
        for worker in (NEW / task).glob("*.sbatch"):
            s = worker.read_text()
            assert "scripts/adaptive/task" not in s
            assert "scripts/research/fusion_image/" in s
            assert "SLURM_SUBMIT_DIR" in s


def test_task1_new_controller_import_and_job_path(tmp_path):
    """Validate same AL policy & controlled scheduler paths without launching sbatch."""
    env = dict(os.environ, AL_PIPELINE_REPO=str(ROOT))
    program = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'scripts/research/fusion_image/task1'))
import task1_closed_loop as al
import task1_schedule as schedule
assert al.CYCLES == 9
assert len(al.CONFIGS) == 3
assert set(al.PHASE1) == {'random','entropy'}
assert set(al.PHASE2) == {'entropy_diversity','entropy_balanced_explore'}
assert al.REPO == Path.cwd()
assert len(al.sha_sources(Path.cwd())) == len(al.SOURCES)
assert al.SOURCES[0] == 'scripts/research/fusion_image/task1/task1_closed_loop.py'
assert schedule.REPO == al.REPO
assert schedule.verify_plan is al.verify_plan
"""
    out = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
