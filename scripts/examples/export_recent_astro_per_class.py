#!/usr/bin/env python3
"""Create a small *read-only* per-class validation export from saved runs.

Instructor-only preparation. Reads existing AstroMiNN object-level evaluation
metrics; it NEVER loads images, checkpoints, or prediction arrays and does not
execute inference. No test metrics or object identifiers are exported.

Example:
  python scripts/examples/export_recent_astro_per_class.py \
    --validation-csv examples/data/astro_recent_validation.csv \
    --artifact-root /work/hdd/bcrv/$USER/applecider_adaptive \
    --output ~/astro_recent_per_class_pilot.zip

The resulting ZIP must be reviewed before publishing in teaching materials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

CLASSES = ("AGN", "TDE", "SN I", "SN II", "CV")
CSV_COLUMNS = (
    "suite", "strategy", "seed", "cycle", "class_name",
    "precision", "recall", "f1", "support", "n_acquired_total",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_validation(path: Path) -> dict[tuple[str, str, int, int], tuple[float, int]]:
    """Choose one independent object-level macro-F1 value per recorded cycle."""
    baseline = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "suite", "strategy", "seed", "cycle", "split", "metric",
            "value", "n_acquired_total",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Validation history is missing required columns")
        for row in reader:
            if row["split"] != "validation" or row["metric"] != "f1_macro":
                continue
            key = (row["suite"], row["strategy"], int(row["seed"]), int(row["cycle"]))
            if key in baseline:
                raise ValueError(f"Duplicate history entry for {key}")
            value = float(row["value"])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid validation macro-F1 for {key}")
            n_acquired = int(row["n_acquired_total"])
            if n_acquired < 0:
                raise ValueError(f"Negative acquisition budget for {key}")
            baseline[key] = (value, n_acquired)
    if not baseline:
        raise ValueError("No validation macro-F1 records found")
    grouped = defaultdict(set)
    for suite, policy, seed, cycle in baseline:
        grouped[(suite, policy, seed)].add(cycle)
    for run, cycles in grouped.items():
        if cycles != set(range(10)):
            raise ValueError(f"Expected cycles 0..9 for {run}, found {sorted(cycles)}")
    return baseline


def _load_per_class(path: Path, *, key: tuple[str, str, int, int]) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or unreadable per-class metrics at {key}: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("object"), dict):
        raise ValueError(f"Expected object-level per-class metrics at {key}: {path}")
    data = payload["object"]
    if set(data) != set(CLASSES):
        raise ValueError(f"Unexpected class names for {key}: {sorted(data)}")
    result = {}
    for name in CLASSES:
        raw = data[name]
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid class metrics for {key}/{name}")
        vals = {}
        for field in ("precision", "recall", "f1"):
            val = float(raw[field])
            if not math.isfinite(val) or not 0 <= val <= 1:
                raise ValueError(f"Invalid {field} for {key}/{name}: {val}")
            vals[field] = val
        support = raw["support"]
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise ValueError(f"Invalid support for {key}/{name}: {support!r}")
        vals["support"] = support
        result[name] = vals
    return result


def collect(*, validation_csv: Path, artifact_root: Path, tolerance=2e-6):
    """Verify archived per-class F1 against *already distributed* macro-F1."""
    baseline = _read_validation(validation_csv)
    records = []
    hashes = []
    supports_by_run = {}
    for key in sorted(baseline):
        suite, policy, seed, cycle = key
        aggregate_f1, budget = baseline[key]
        source = (artifact_root / "runs" / suite / policy / f"seed_{seed}"
                  / f"cycle_{cycle:03d}" / "evaluation" / "validation_per_class_metrics.json")
        per_class = _load_per_class(source, key=key)
        measured = sum(per_class[name]["f1"] for name in CLASSES) / len(CLASSES)
        if abs(measured - aggregate_f1) > tolerance:
            raise ValueError(
                f"Per-class F1 does not reproduce recorded validation macro-F1 "
                f"at {key}: {measured:.8f} vs {aggregate_f1:.8f}"
            )
        support_counts = tuple(per_class[name]["support"] for name in CLASSES)
        run_key = key[:3]
        if run_key in supports_by_run and supports_by_run[run_key] != support_counts:
            raise ValueError(f"Validation class supports change with cycle for {run_key}")
        supports_by_run[run_key] = support_counts
        hashes.append(_sha256(source))
        for name in CLASSES:
            records.append({
                "suite": suite, "strategy": policy, "seed": seed, "cycle": cycle,
                "class_name": name, "n_acquired_total": budget,
                **per_class[name],
            })
    if len(records) != len(baseline) * len(CLASSES):
        raise AssertionError("Partial class export")
    provenance = {
        "schema": "astro_recent_object_class_validation_v1",
        "description": "Recorded, already-computed validation metrics at object level",
        "split": "validation",
        "classes": list(CLASSES),
        "runs": len(supports_by_run),
        "cycles_per_run": 10,
        "class_rows": len(records),
        "macro_f1_checked": len(baseline),
        "baseline_history_sha256": _sha256(validation_csv),
        "source_digest_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
        "included": ["object-level validation precision", "recall", "f1", "support"],
        "excluded": ["test data", "object IDs", "individual predictions", "sample images", "model checkpoints"],
        "note": "Saved metrics were calculated during previous model evaluations; no inference performed.",
    }
    return records, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path,
                        default=Path("examples/data/astro_recent_validation.csv"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists():
        parser.error(f"Refusing to overwrite {out}")
    records, provenance = collect(
        validation_csv=args.validation_csv.expanduser(),
        artifact_root=args.artifact_root.expanduser(),
    )
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(out, "x", ZIP_DEFLATED) as z:
        z.writestr("astro_recent_per_class.csv", buf.getvalue())
        z.writestr("astro_recent_per_class_provenance.json",
                   json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archive": str(out), "runs": provenance["runs"],
                      "class_rows": provenance["class_rows"],
                      "validated_cycles": provenance["macro_f1_checked"]}, indent=2))
    print("Read-only export complete. Review before committing teaching data.")


if __name__ == "__main__":
    main()