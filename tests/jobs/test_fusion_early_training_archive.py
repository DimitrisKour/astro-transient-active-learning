"""Check archival separation of old fusion training/cache workers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "scripts/archive/fusion_image"
ACTIVE = ROOT / "scripts/adaptive"

GROUPS = {
    "recipe_v1": ("fusion_pre_task0.py", "fusion_pre_task0.sbatch",
                  "submit_fusion_pre_task0.sh"),
    "fastcache_v2": ("fusion_pre_task0_fast.py", "fusion_pre_task0_fast.sbatch",
                     "submit_fusion_pre_task0_fast.sh"),
    "early_diagnostics": ("diagnose_fusion_pre_task0.sh",
                          "fusion_fastcache_cpu.sbatch",
                          "fusion_fastcache_cpu_safe.sbatch",
                          "fusion_first_batch_probe.py",
                          "fusion_first_batch_probe.sbatch"),
}


def test_old_training_launchers_are_archived_with_internal_links():
    assert (ARCHIVE / "README.md").is_file()
    names = {name: group for group, files in GROUPS.items() for name in files}
    assert len(names) == 11
    for name, group in names.items():
        path = ARCHIVE / group / name
        assert path.is_file(), str(path)
        assert not (ACTIVE / name).exists(), name
        text = path.read_text(encoding="utf-8")
        for dep in names:
            assert f"scripts/adaptive/{dep}" not in text, (path, dep)


def test_full_history_and_resident_recipe_remain_active():
    for name in (
        "fusion_full_historical_cache_v1.py",
        "fusion_pre_task0_resident.py",
        "fusion_pre_task0_resident.sbatch",
        "submit_fusion_pre_task0_resident.sh",
        "fusion_pre_task0_extensions.py",
        "task0_full_history.py",
        "task1_closed_loop.py",
        "task1_schedule.py",
    ):
        assert (ACTIVE / name).is_file(), name
