"""Check that obsolete L0 launchers are archived, without executing any HPC job."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "scripts/archive/fusion_image/early_l0"
LEGACY = ROOT / "scripts/adaptive"
FILES = (
    "fusion_cache_v2.py",
    "fusion_cache_v2_benchmark.sbatch",
    "fusion_cache_v2_finalize.sbatch",
    "fusion_cache_v2_shard.sbatch",
    "submit_fusion_cache_v3.sh",
    "fusion_l0_matrix.py",
    "fusion_l0_gpu.sbatch",
    "fusion_l0_report.sbatch",
    "submit_fusion_l0_matrix.sh",
    "fusion_cached_l0_benchmark_v2.py",
    "fusion_cached_l0_full_seed135.py",
    "fusion_cached_l0_full_seed135.sbatch",
)

def test_early_l0_files_archived_and_not_in_default_adaptive_folder():
    assert (ARCHIVE / "README.md").is_file()
    assert all((ARCHIVE / f).is_file() for f in FILES)
    assert all(not (LEGACY / f).exists() for f in FILES)


def test_archived_scripts_use_archived_references_and_current_research_remains():
    for name in FILES:
        path = ARCHIVE / name
        text = path.read_text(encoding="utf-8")
        for other in FILES:
            assert f"scripts/adaptive/{other}" not in text, (name, other)

    matrix = (ARCHIVE / "fusion_l0_matrix.py").read_text(encoding="utf-8")
    assert "scripts/archive/fusion_image/early_l0/fusion_l0_matrix.py" in matrix
    for name in (
        "fusion_full_historical_cache_v1.py",
        "fusion_pre_task0_resident.py",
        "task0_full_history.py",
        "task1_closed_loop.py",
        "task1_schedule.py",
    ):
        assert (LEGACY / name).is_file(), name
