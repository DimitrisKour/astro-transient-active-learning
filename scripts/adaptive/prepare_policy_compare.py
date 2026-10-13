#!/usr/bin/env python
"""Route writes to HDD/projects and freeze the submitted source snapshot."""

import argparse
import hashlib
import json
import re
import tempfile
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path,
                        help="Reuse the original replay registry instead of a local config.")
    parser.add_argument("--verify-source", action="store_true")
    args = parser.parse_args()
    repo, job_dir = args.repo.resolve(), args.job_dir.resolve()
    persistent = Path("/projects/bcrv/dkourtesis/applecider_adaptive")
    data = Path("/work/hdd/bcrv/dkourtesis/applecider_adaptive")
    if not job_dir.is_relative_to(persistent):
        raise ValueError("Job products must be under the persistent experiment root.")
    source = args.source_registry or repo / "configs/delta.toml"
    content = source.read_text()
    original = tomllib.loads(content)
    match = re.search(r"(?ms)^\[storage\]\s*\n(.*?)(?=^\[|\Z)", content)
    if match is None:
        raise ValueError("Registry has no [storage] section.")
    section = match.group(0)
    for key, value in [("persistent_root", persistent), ("data_root", data)]:
        section, count = re.subn(rf"(?m)^\s*{key}\s*=.*$", f'{key} = "{value}"', section)
        if count != 1:
            raise ValueError(f"Expected exactly one storage {key}.")
    routed = content[: match.start()] + section + content[match.end() :]
    updated = tomllib.loads(routed)
    assert {k: v for k, v in original.items() if k != "storage"} == {
        k: v for k, v in updated.items() if k != "storage"
    }
    for path in (persistent, data, job_dir, job_dir / "logs"):
        path.mkdir(parents=True, exist_ok=True)
        if not (path.resolve().is_relative_to(persistent) or path.resolve().is_relative_to(data)):
            raise ValueError(f"Output directory resolves outside its allowed root: {path}")
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b"0" * 1048576)
            probe.flush()
            import os

            os.fsync(probe.fileno())
        print("Write check OK:", path)
    registry = job_dir / "registry_hdd.toml"
    if registry.exists():
        if registry.read_text() != routed:
            raise ValueError("Existing suite registry differs; choose a new suite ID.")
    else:
        registry.write_text(routed)
    files = [
        *sorted((repo / "src/al_pipeline").rglob("*.py")),
        *sorted((repo / "scripts/adaptive").glob("*policy*")),
        repo / "scripts/adaptive/historical_final_cycle.sbatch",
        repo / "scripts/run_astrominn_historical_replay.py",
    ]
    hashes = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.is_file()
    }
    snapshot = job_dir / "source_snapshot.json"
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != hashes:
            raise ValueError("Code changed after suite preparation; use a new suite ID.")
    elif args.verify_source:
        raise ValueError("Source snapshot is missing; run submission preparation first.")
    else:
        snapshot.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print("Registry:", registry)
    print("Historical input paths unchanged; large outputs routed to HDD.")


if __name__ == "__main__":
    main()
