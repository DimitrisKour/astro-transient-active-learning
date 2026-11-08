#!/usr/bin/env python3
"""One-time, audited Task11 v1.1 *packaging-only* dependency repair.

Scope: only the frozen Task11 advanced_acquisition/__init__.py. The Task11
experimental suite, scientific plan, Task4 source and all acquisition algorithms
are immutable. May run only before pilots or cycle training begin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SUITE = "astrominn_task11_advanced_acquisition_v1"
JOB = Path("/projects/bcrv/dkourtesis/applecider_adaptive/closed_loop") / SUITE
REL = "src/al_pipeline/core/advanced_acquisition/__init__.py"
IMPORT = """from al_pipeline.core.advanced_acquisition.ensemble_disagreement import (
    EnsembleDisagreementScorer,
)
"""
EXPORT = '    "EnsembleDisagreementScorer",\n'
REPAIR_NAME = "import_dependency_repair_v1_1.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_atomic(path: Path, content) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def all_source_hashes(source: Path) -> dict[str, str]:
    return {
        str(p.relative_to(source)): digest(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".toml")
    }


def repaired(text: str) -> str:
    if text.count(IMPORT) != 1 or text.count(EXPORT) != 1:
        raise ValueError("Unexpected advanced acquisition package exports; no changes made")
    updated = text.replace(IMPORT, "", 1).replace(EXPORT, "", 1)
    if "ensemble_disagreement" in updated or "EnsembleDisagreementScorer" in updated:
        raise ValueError("Unexpected remaining ensemble imports; no changes made")
    if "BadgeInput" not in updated or "CoreSetInput" not in updated:
        raise ValueError("Task11 BADGE/core-set imports missing; no changes made")
    return updated


def check_no_experiment_started(job: Path) -> None:
    if (job / "submitted.json").exists():
        raise RuntimeError("Task11 has been submitted; refusing frozen-source repair")
    pilots = job / "pilot"
    if pilots.exists() and any(pilots.rglob("result.json")):
        raise RuntimeError("A Task11 acquisition pilot already completed; refusing repair")
    if pilots.exists() and any(pilots.rglob("*.npz")):
        raise RuntimeError("A Task11 pilot has acquisition artifacts; refusing repair")
    persistent_runs = job.parent.parent / "runs" / SUITE
    if persistent_runs.exists() and any(persistent_runs.rglob("run_state.json")):
        raise RuntimeError("Task11 training state exists; refusing repair")


def repair(job: Path) -> dict:
    job = Path(job).resolve()
    source = job / "source"
    marker = job / REPAIR_NAME
    manifest_path = job / "frozen_manifest.json"
    plan_path = job / "scientific_plan.json"
    target = source / REL
    if not (source.is_dir() and manifest_path.is_file() and plan_path.is_file() and target.is_file()):
        raise FileNotFoundError("Prepared Task11 frozen source/manifest/plan missing")
    plan = json_read(plan_path)
    if plan.get("suite") != SUITE or plan.get("fits") != 81 or plan.get("policies") != ["badge", "core_set", "fisher_batch"]:
        raise ValueError("Unexpected Task11 scientific plan; refusing repair")
    manifest = json_read(manifest_path)
    if marker.exists():
        recorded = json_read(marker)
        if recorded.get("change") != REL or recorded.get("updated_sha256") != digest(target):
            raise ValueError("Existing repair marker conflicts with frozen source")
        if manifest.get(REL) != digest(target) or all_source_hashes(source) != manifest:
            raise ValueError("Frozen source differs from repaired manifest")
        print("Task11 import repair v1.1 already present and verified")
        return recorded
    if all_source_hashes(source) != manifest:
        raise ValueError("Frozen source hash mismatch: cannot safely repair")
    check_no_experiment_started(job)
    original = target.read_text(encoding="utf-8")
    modified = repaired(original)
    old = digest(target)
    if manifest.get(REL) != old:
        raise ValueError("Target hash disagrees with frozen manifest")
    record = {
        "repair_version": "1.1", "kind": "packaging_import_only",
        "change": REL, "original_sha256": old,
        "updated_sha256": hashlib.sha256(modified.encode("utf-8")).hexdigest(),
        "scientific_plan_sha256": digest(plan_path),
        "reason": "Remove unused ensemble-disagreement import requiring missing core.prediction package",
        "scientific_changes": False,
    }
    # Save a byte-for-byte copy for auditing and potential manual reconstruction.
    backup = job / "repairs" / "advanced_acquisition_init.original.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise RuntimeError("Unexpected existing source backup; inspect before repair")
    backup.write_bytes(target.read_bytes())
    next_path = target.with_name(target.name + ".task11new")
    next_path.write_text(modified, encoding="utf-8")
    if digest(next_path) != record["updated_sha256"]:
        raise ValueError("Staged updated source hash mismatch")
    next_path.replace(target)
    new_manifest = dict(manifest)
    new_manifest[REL] = digest(target)
    json_atomic(manifest_path, new_manifest)
    json_atomic(marker, record)
    if all_source_hashes(source) != new_manifest:
        raise ValueError("Post-repair frozen source verification failed")
    print("Task11 import repair v1.1 applied; source hash manifest updated; scientific plan unchanged")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, default=JOB)
    args = parser.parse_args()
    print(json.dumps(repair(args.job_dir), indent=2))


if __name__ == "__main__":
    main()
