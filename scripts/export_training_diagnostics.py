#!/usr/bin/env python
"""Export small diagnostic metadata/results, excluding models and raw sample arrays."""

import argparse
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-logs", action="store_true")
    args = parser.parse_args()
    root = args.plan.resolve().parent
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [root / name for name in (
        "plan.json", "plan.sha256", "preflight.json", "submissions.tsv",
    )]
    for directory in ("tasks", "report") + (("logs",) if args.include_logs else ()):
        parent = root / directory
        if parent.exists():
            paths.extend(p for p in parent.rglob("*") if p.is_file())
    paths = [p for p in paths if p.exists() and p.suffix not in {".pth", ".npy", ".npz"}]
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in sorted(set(paths)):
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"Export path escapes the diagnostic directory: {path}")
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Exported {len(paths)} small files to {output}; model weights and arrays excluded.")


if __name__ == "__main__":
    main()
