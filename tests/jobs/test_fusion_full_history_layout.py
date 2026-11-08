"""Confirm that the current full-historical cache/recipe utilities are grouped."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "scripts/research/fusion_image"
OLD = ROOT / "scripts/adaptive"
STAGES = {
    "cache": (
        "fusion_full_historical_cache_v1.py",
        "fusion_full_historical_cache_shard.sbatch",
        "fusion_full_historical_cache_finalize.sbatch",
        "submit_fusion_full_historical_cache_v1.sh",
        "fusion_fastcache.py",
        "fusion_fastcache_cpu_hdd.sbatch",
    ),
    "recipe": (
        "fusion_pre_task0_resident.py",
        "fusion_pre_task0_resident.sbatch",
        "submit_fusion_pre_task0_resident.sh",
        "fusion_resident_probe.py",
        "fusion_resident_probe.sbatch",
        "fusion_pre_task0_extensions.py",
        "fusion_pre_task0_extensions.sbatch",
        "submit_fusion_pre_task0_extensions.sh",
    ),
}


def test_grouped_full_historical_stages_and_no_stale_launcher_paths():
    assert (RESEARCH / "README.md").exists()
    for folder, names in STAGES.items():
        for name in names:
            path = RESEARCH / folder / name
            assert path.is_file(), path
            assert not (OLD / name).exists(), name
            code = path.read_text(encoding="utf-8")
            for other_folder, other_names in STAGES.items():
                for dep in other_names:
                    assert f"scripts/adaptive/{dep}" not in code, (path, dep)


def test_task0_and_task1_untouched_until_independent_migration():
    for name in (
        "task0_full_history.py",
        "task0_full_history.sbatch",
        "submit_task0_full_history.sh",
        "task1_closed_loop.py",
        "task1_schedule.py",
        "task1_cycle.sbatch",
        "submit_task1_closed_loop.sh",
    ):
        assert (OLD / name).is_file(), name
