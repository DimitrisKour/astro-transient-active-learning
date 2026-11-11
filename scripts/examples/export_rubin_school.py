#!/usr/bin/env python3
"""Export independently runnable Day 3 materials to a Rubin ML Training School checkout.

From this repository's root:
    python scripts/examples/export_rubin_school.py --school-root /path/to/Rubin-ML-Training-School

No dependencies beyond Python's standard library are required to *export*.
The exported student notebooks are executable with al-pipeline[teaching].
This never commits, pushes, or changes the upstream school repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
RELATIVE_FOLDER = Path("day3/01_active_learning")
SOURCE_NOTEBOOKS = {
    "01_active_learning_digits.ipynb": SOURCE_ROOT / "examples/01_active_learning_digits.ipynb",
    "02_applecider_astronomy_active_learning.ipynb": (
        SOURCE_ROOT / "examples/02_applecider_astronomy_active_learning.ipynb"
    ),
}
DATA_FILES = (
    "astro_recent_validation.csv",
    "astro_recent_per_class.csv",
    "astro_cycle2_overlap_by_seed.csv",
    "astro_cycle2_overlap_mean.csv",
    "astro_recent_provenance.json",
    "astro_recent_per_class_provenance.json",
    "astro_cycle2_overlap_provenance.json",
)
DATA_SOURCE = SOURCE_ROOT / "examples/data"
RUNNER_SOURCE = SOURCE_ROOT / "scripts/examples/school_templates/run_notebooks.py"

# This code must remain executable whether the notebook is launched at the
# school root or from its Day 3 folder. The school has a top-level data/ ignore
# rule; teaching_data/ is intentionally used for tracked, non-sensitive CSVs.
OLD_DATA_DISCOVERY = (
    "DATA_DIR = next((p for p in (Path('examples/data'), Path('data'))\n"
    "                 if (p / 'astro_recent_validation.csv').is_file()), None)\n"
    "if DATA_DIR is None:\n"
    "    raise FileNotFoundError('Missing bundled examples/data/astro_recent_validation.csv')"
)
NEW_DATA_DISCOVERY = (
    "DATA_DIR = next(\n"
    "    (p for p in (\n"
    "        Path('teaching_data'),\n"
    "        Path('day3/01_active_learning/teaching_data'),\n"
    "    ) if (p / 'astro_recent_validation.csv').is_file()), None\n"
    ")\n"
    "if DATA_DIR is None:\n"
    "    raise FileNotFoundError(\n"
    "        'Missing bundled teaching_data/astro_recent_validation.csv; '\n"
    "        'run from the school root or day3/01_active_learning/'\n"
    "    )"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def export(school_root: Path, *, overwrite: bool) -> Path:
    school_root = school_root.expanduser().resolve()
    if not (school_root / ".git").exists() or not (school_root / "day3").is_dir():
        raise ValueError(
            f"{school_root} is not a cloned Rubin-ML-Training-School checkout"
        )
    dest = school_root / RELATIVE_FOLDER
    if not dest.is_dir():
        raise ValueError(f"Expected session folder not found: {dest}")
    if not (dest / "README.md").is_file():
        raise ValueError("Expected the existing school session README.md")

    copied = {
        dest / filename: src for filename, src in SOURCE_NOTEBOOKS.items()
    }
    copied.update({
        dest / "teaching_data" / filename: DATA_SOURCE / filename
        for filename in DATA_FILES
    })
    copied[dest / "teaching_data/README.md"] = (SOURCE_ROOT / "scripts/examples/school_templates/teaching_data_README.md")
    copied[dest / "run_notebooks.py"] = RUNNER_SOURCE

    readme_path = dest / "README.md"
    school_readme = (SOURCE_ROOT / "scripts/examples/school_templates/README.md").read_text(
        encoding="utf-8"
    )
    metadata_path = dest / "source_manifest.json"
    extra_targets = (readme_path, metadata_path)
    occupied = [str(path) for path in (*copied.keys(), *extra_targets)
                if path.exists()]
    legacy_license = dest / "LICENSE_AL_PIPELINE"
    if legacy_license.is_file() and overwrite:
        # Remove only the exact notice produced by older exports.
        legacy_mit_sha256 = "b1bf2d2d2033c91410bdb50ca0feba6798fbbf7605c2947f26977a42341d3bb0"
        if sha256(legacy_license) == legacy_mit_sha256:
            legacy_license.unlink()
    if occupied and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing files (pass --overwrite after review): "
            + ", ".join(occupied)
        )
    for path, src in copied.items():
        if not src.is_file():
            raise FileNotFoundError(src)
    for path, src in copied.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".ipynb":
            notebook = json.loads(src.read_text(encoding="utf-8"))
            if path.name.startswith("02_"):
                found = 0
                for cell in notebook.get("cells", []):
                    if cell.get("cell_type") != "code":
                        continue
                    parts = cell.get("source", [])
                    source = "".join(parts) if isinstance(parts, list) else parts
                    if OLD_DATA_DISCOVERY in source:
                        source = source.replace(OLD_DATA_DISCOVERY, NEW_DATA_DISCOVERY)
                        cell["source"] = source.splitlines(keepends=True)
                        found += 1
                if found != 1:
                    raise RuntimeError(
                        f"Expected one original astronomy data loader, found {found}; "
                        "update the exporter before distributing a changed source notebook."
                    )
            notebook.setdefault("metadata", {}).setdefault("school_export", {})
            notebook["metadata"]["school_export"].update({
                "source_repository": "DimitrisKour/astro-transient-active-learning",
                "original_path": str(src.relative_to(SOURCE_ROOT)),
                "source_commit": source_commit(),
                "reproducibility": "All cells executable after the documented pip installation",
            })
            path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copyfile(src, path)
    readme_path.write_text(
        school_readme.replace("__SOURCE_COMMIT__", source_commit()), encoding="utf-8"
    )

    manifest = {
        "source_repository": "https://github.com/DimitrisKour/astro-transient-active-learning",
        "source_commit": source_commit(),
        "exported_folder": RELATIVE_FOLDER.as_posix(),
        "content": {
            str(path.relative_to(dest)): sha256(path)
            for path in sorted(copied)
        },
        "research_data_note": (
            "Only recorded validation summaries and aggregate acquisition "
            "overlaps are distributed; no raw observations, checkpoint files, "
            "object identifiers or test predictions."
        ),
    }
    manifest["content"]["README.md"] = sha256(readme_path)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"Exported {len(copied) + 2} files to {dest}")
    print(f"Source commit: {manifest['source_commit']}")
    print("Run: python -m pip install 'al-pipeline[teaching] @ git+https://github.com/"
          "DimitrisKour/astro-transient-active-learning.git@"
          + manifest["source_commit"] + "'")
    print("Then: python day3/01_active_learning/run_notebooks.py")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--school-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace only this session's exported teaching files and README",
    )
    args = parser.parse_args()
    export(args.school_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
