"""Checks for a platform-neutral astronomy teaching export entry point."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/examples/prepare_astro_school_exports.py"


def test_registry_wrapper_exports_features_and_discovers_reports(tmp_path):
    data = tmp_path / "samples"
    data.mkdir()
    manifests = {}
    labels = ("AGN", "TDE", "SN Ia", "SN IIn", "CV")
    rng = np.random.default_rng(7)
    for split in ("train", "validation", "test"):
        manifest = tmp_path / f"{split}.csv"
        manifests[split] = manifest
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("obj_id", "target", "alert", "filepath"))
            writer.writeheader()
            for cls in labels:
                for i in range(5):
                    oid = f"{split}_{cls}_{i}"
                    path = data / f"{oid}.npy"
                    np.save(path, {
                        "obj_id": oid, "alerte": 2,
                        "metadata": rng.normal(size=34).astype('float32'),
                        "image": rng.normal(size=(3, 63, 63)).astype('float32'),
                    }, allow_pickle=True)
                    writer.writerow({"obj_id":oid, "target":cls, "alert":2, "filepath":path.name})
    registry = tmp_path / "registry.toml"
    registry.write_text('''[datasets.astrominn_ztf]
train_manifest = "''' + str(manifests['train']) + '''"
validation_manifest = "''' + str(manifests['validation']) + '''"
test_manifest = "''' + str(manifests['test']) + '''"
data_root = "''' + str(data) + '''"
object_id_column = "obj_id"
sample_file_column = "filepath"
label_column = "target"
''')
    reports = tmp_path / "reports" / "suite1"
    reports.mkdir(parents=True)
    (reports / "evaluation_records.csv").write_text("suite,initialization,strategy,seed,cycle,split,metric,value,n_labeled,n_acquired_total\nB,warm_start,entropy,135,0,validation,f1_macro,0.6,25,0\n")
    output = tmp_path / "out"
    result = subprocess.run([sys.executable, str(SCRIPT), "--registry", str(registry),
                             "--output-directory", str(output),
                             "--train-objects", "25", "--validation-objects", "25",
                             "--test-objects", "25", "--report-root", str(tmp_path / "reports")],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert (output / "astro_real_v1.npz").is_file()
    assert str(reports) in result.stdout
    assert "Part B not exported" in result.stdout
    assert not (output / "astro_historical_evaluation.csv").exists()
    second = subprocess.run([sys.executable, str(SCRIPT), "--registry", str(registry),
                             "--output-directory", str(output), "--skip-features",
                             "--report-directory", str(reports)],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert (output / "astro_historical_evaluation.csv").exists()
    assert "Exporting the explicitly selected archived report" in second.stdout