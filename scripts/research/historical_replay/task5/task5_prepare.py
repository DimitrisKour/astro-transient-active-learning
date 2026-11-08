#!/usr/bin/env python3
"""Create a frozen, reproducible Task-5 source and routed registry.

Never mutates either Experiment B or Task 1/2/3 outputs. Repeated preparation
checks that nothing already frozen has changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path


def files_to_freeze(repo: Path) -> list[Path]:
    return [
        *sorted((repo / "src" / "al_pipeline").rglob("*.py")),
        repo / "scripts" / "run_astrominn_closed_loop_restart.py",
        *sorted((repo / "scripts" / "research" / "historical_replay" / "task5").glob("task5_*.py")),
        *sorted((repo / "scripts" / "research" / "historical_replay" / "task5").glob("task5_*.sh")),
        *sorted((repo / "scripts" / "research" / "historical_replay" / "task5").glob("task5_*.sbatch")),
    ]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def routed_registry(text: str, *, persistent: Path, data: Path) -> str:
    orig = tomllib.loads(text)
    m = re.search(r"(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)", text)
    if m is None:
        raise ValueError("Missing [storage] section in registry")
    section = m.group(0)
    for key, value in (("persistent_root", persistent), ("data_root", data)):
        section, count = re.subn(rf"(?m)^\s*{key}\s*=.*$", f'{key} = "{value}"', section)
        if count != 1:
            raise ValueError(f"Registry requires one storage.{key}")
    result = text[:m.start()] + section + text[m.end():]
    routed = tomllib.loads(result)
    assert {k:v for k,v in orig.items() if k != "storage"} == {
        k:v for k,v in routed.items() if k != "storage"
    }
    return result


def prepare(*, repo: Path, job_dir: Path, source_registry: Path,
            persistent: Path, data: Path) -> dict[str, str]:
    repo, job_dir = repo.resolve(), job_dir.resolve()
    if not job_dir.is_relative_to(persistent.resolve()):
        raise ValueError("Task5 job directory must be inside persistent storage")
    from_file = repo / "scripts" / "research" / "historical_replay" / "task5" / "task5_build_runner.py"
    subprocess.run(["python", str(from_file), "--repo", str(repo)], check=True)
    files = files_to_freeze(repo)
    if not files or not all(p.is_file() for p in files):
        raise FileNotFoundError("Task 5 source files missing")
    names = [str(p.relative_to(repo)) for p in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate source paths")
    source = job_dir / "source"
    manifest = {name: digest(file) for name, file in zip(names, files, strict=True)}
    snapshot = job_dir / "source_snapshot.json"
    registry = job_dir / "registry.toml"
    routed = routed_registry(source_registry.read_text(encoding="utf-8"),
                             persistent=persistent, data=data)
    for path in (persistent, data, job_dir, job_dir / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != manifest:
            raise ValueError("Source changed after freezing Task 5; use new suite ID")
        if not registry.exists() or registry.read_text() != routed:
            raise ValueError("Registry changed after freezing Task 5")
        verify_source(job_dir)
        return manifest
    if source.exists() or registry.exists():
        raise ValueError("Incomplete previous preparation; inspect before continuing")
    source.mkdir()
    for path, name in zip(files, names, strict=True):
        dst = source / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
    verify_expected = {name: digest(source/name) for name in names}
    if verify_expected != manifest:
        raise RuntimeError("Frozen source differs from original during copy")
    registry.write_text(routed, encoding="utf-8")
    snapshot.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return manifest


def verify_source(job_dir: Path) -> None:
    job_dir = Path(job_dir)
    manifest = json.loads((job_dir / "source_snapshot.json").read_text())
    source = job_dir / "source"
    for name, expected in manifest.items():
        p = source / name
        if not p.is_file() or digest(p) != expected:
            raise ValueError(f"Frozen Task 5 source mismatch: {p}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--job-dir", type=Path, required=True)
    p.add_argument("--source-registry", type=Path, required=True)
    p.add_argument("--persistent", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.verify_only:
        verify_source(args.job_dir)
        if not (args.job_dir / "registry.toml").is_file():
            raise FileNotFoundError("Missing frozen registry")
        print("Frozen Task 5 source verified")
    else:
        result = prepare(repo=args.repo, job_dir=args.job_dir,
                         source_registry=args.source_registry,
                         persistent=args.persistent, data=args.data)
        print(f"Frozen Task 5 source: {len(result)} files -> {args.job_dir/'source'}")


if __name__ == "__main__":
    main()
