"""CPU-only tests; synthetic inputs only, no access to Delta data needed."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from al_pipeline.astronomy.representation.astrominn import FeatureRecorder, extract_features
from al_pipeline.astronomy.representation.core import (
    digest_file, digest_object, fit_and_evaluate, latest_rows, shard_for,
    stratified_subset, validate_disjoint, write_json_atomic, write_npz_atomic,
)
from al_pipeline.astronomy.representation.design import (
    RepresentationAdaptationDesign, RepresentationClosedLoopDesign,
)


def manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, ["obj_id", "class", "alert", "sample_path"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_latest_only_and_provenance(tmp_path):
    path = manifest(tmp_path, [
        {"obj_id": "o1", "class": "A", "alert": "0", "sample_path": "a.npy"},
        {"obj_id": "o2", "class": "B", "alert": "2", "sample_path": "b.npy"},
        {"obj_id": "o1", "class": "A", "alert": "4", "sample_path": "c.npy"},
    ])
    rows = latest_rows(path, "obj_id", "class", {"A": 0, "B": 1})
    assert rows == [
        {"object_id": "o1", "target": 0, "alert": 4, "row": 2},
        {"object_id": "o2", "target": 1, "alert": 2, "row": 1},
    ]


def test_reject_conflicting_labels(tmp_path):
    path = manifest(tmp_path, [
        {"obj_id": "o1", "class": "A", "alert": 0},
        {"obj_id": "o1", "class": "B", "alert": 1},
    ])
    with pytest.raises(ValueError, match="Conflicting labels"):
        latest_rows(path, "obj_id", "class", {"A": 0, "B": 1})


def test_reject_duplicate_alert(tmp_path):
    path = manifest(tmp_path, [
        {"obj_id": "o1", "class": "A", "alert": 0},
        {"obj_id": "o1", "class": "A", "alert": 0},
    ])
    with pytest.raises(ValueError, match="Duplicate alert"):
        latest_rows(path, "obj_id", "class", {"A": 0})


def test_object_overlap_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint({"train": [{"object_id": "X"}], "test": [{"object_id": "X"}]})


def test_stratified_selection_deterministic():
    records = [{"object_id": f"{cls}_{i}", "target": cls} for cls in range(5) for i in range(10)]
    a = stratified_subset(records, 3)
    b = stratified_subset(list(reversed(records)), 3)
    assert a == b
    assert len(a) == 15
    assert {row["target"] for row in a} == set(range(5))


def test_shards_exclusive_and_stable():
    objects = [f"ZTF-{i}" for i in range(101)]
    for obj in objects:
        assert sum(shard_for(obj, 13) == i for i in range(13)) == 1
    assert [shard_for(x, 13) for x in objects] == [shard_for(x, 13) for x in objects]


class Tower(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.proj = nn.Linear(inp, out)

    def forward(self, x):
        return self.proj(x)


class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_out = nn.Linear(6, 5)

    def forward(self, x):
        return self.fc_out(x)


class MockAstroMiNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lc_tower = Tower(3, 2)
        self.spatial_tower = Tower(2, 2)
        self.coord_tower = Tower(3, 2)
        self.vit_tower = MockViT()

    def forward(self, metadata, *, image):
        x = torch.cat([self.lc_tower(metadata), self.spatial_tower(image),
                       self.coord_tower(metadata)], dim=1)
        return {"logits": self.vit_tower(x)}


class Samples(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, i):
        return {"metadata": torch.tensor([i, i + 1, i + 2], dtype=torch.float32),
                "image": torch.tensor([i + 1, i - 1], dtype=torch.float32),
                "obj_id": str(i), "target_index": i % 5, "alert": i}


def test_capture_matches_classifier_input_and_all_towers():
    model = MockAstroMiNN()
    with torch.no_grad():
        result = extract_features(model, DataLoader(Samples(), batch_size=3),
                                  representations=("vit_cls", "lc_expert", "expert_concat"))
    assert result["vit_cls"].shape == (7, 6)
    assert result["expert_concat"].shape == (7, 6)
    np.testing.assert_allclose(result["vit_cls"], result["expert_concat"])
    assert list(result["object_ids"]) == [str(i) for i in range(7)]
    assert len(model.vit_tower.fc_out._forward_pre_hooks) == 0
    assert len(model.lc_tower._forward_hooks) == 0


def test_probe_all_five_classes_and_frozen_hyperparameters():
    rng = np.random.default_rng(135)
    train_y = np.repeat(np.arange(5), 4)
    val_y = np.repeat(np.arange(5), 2)
    test_y = np.repeat(np.arange(5), 3)
    f = lambda labels: np.eye(5)[labels] * 4 + rng.normal(scale=.1, size=(len(labels), 5))
    out = fit_and_evaluate(f(train_y), train_y,
                           {"validation": (f(val_y), val_y), "test": (f(test_y), test_y)},
                           names=("AGN", "TDE", "SN I", "SN II", "CV"))
    assert set(out["splits"]) == {"train", "validation", "test"}
    assert out["configuration"]["fit_data"] == "historical_L0_only"
    assert out["splits"]["validation"]["macro_f1"] > .95
    assert out["splits"]["test"]["objects"] == 15


def test_probe_rejects_missing_class():
    with pytest.raises(ValueError, match="support for every"):
        fit_and_evaluate(np.eye(3), np.array([0, 1, 2]), {}, names=("a", "b", "c", "d", "e"))


def test_atomic_files_and_source_provenance(tmp_path):
    file = tmp_path / "plan.json"
    payload = {"x": [1, 2]}
    write_json_atomic(file, payload)
    assert json.loads(file.read_text()) == payload
    assert digest_file(file) == digest_file(file)
    assert digest_object(payload) != digest_object({"x": [1, 3]})
    npz = tmp_path / "test.npz"
    write_npz_atomic(npz, a=np.arange(3))
    with np.load(npz, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["a"], np.arange(3))


def test_5b_5c_interfaces_only(tmp_path):
    good = RepresentationAdaptationDesign(tmp_path / "a.json", "partial_vit_finetune")
    assert good.fixed_acquisition_manifests
    with pytest.raises(ValueError):
        RepresentationAdaptationDesign(tmp_path / "a.json", "test", fixed_acquisition_manifests=False)
    design = RepresentationClosedLoopDesign(tmp_path / "t4.json", tmp_path / "a.json",
                                            tmp_path / "b.json", "entropy", 473)
    assert design.train_budget == 473


def _cli_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "adaptive" / "task5a.py"
    if not path.exists():
        # The real repo has tests/astronomy/representation; parents[3] is root.
        raise AssertionError(f"CLI absent: {path}")
    spec = importlib.util.spec_from_file_location("task5a_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sharded_report_roundtrip_and_corruption_guard(tmp_path):
    import argparse
    script = _cli_module()
    run_dir = tmp_path / "experiment"
    run_dir.mkdir()
    rng = np.random.default_rng(9)
    sample_counts = {"train": 4, "validation": 2, "test": 2}
    splits = {}
    for split, amount in sample_counts.items():
        rows = [{"object_id": f"{split}-{label}-{i}", "target": label, "alert": i, "row": i}
                for label in range(5) for i in range(amount)]
        splits[split] = {"objects": rows}
    plan = {"schema": "task5a_representation_v1", "mode": "smoke", "seed": 135, "shards": 2,
            "representations": ["vit_cls", "expert_concat"],
            "class_names": ["AGN", "TDE", "SN I", "SN II", "CV"],
            "splits": splits, "probe": {"C": 1.0}, "code_hashes": {},
            "registry": str(tmp_path / "registry"), "registry_sha256": "",
            "checkpoint": str(tmp_path / "checkpoint"), "checkpoint_sha256": "",
            "artifact_dir": str(tmp_path), "artifact_hashes": {}}
    for name in ("registry", "checkpoint"):
        (tmp_path / name).write_text(name)
        plan[f"{name}_sha256"] = digest_file(tmp_path / name)
    for split in splits:
        fn = tmp_path / f"{split}.csv"
        fn.write_text(split)
        splits[split]["manifest"] = str(fn)
        splits[split]["sha256"] = digest_file(fn)
    plan["plan_sha256"] = digest_object(plan)
    write_json_atomic(run_dir / "plan.json", plan)
    for split, entry in splits.items():
        for shard in range(2):
            rows = [r for r in entry["objects"] if shard_for(r["object_id"], 2) == shard]
            dest = run_dir / "features" / split / f"shard_{shard:03d}_of_002.npz"
            labels = np.asarray([r["target"] for r in rows], dtype=np.int64)
            features = np.eye(5)[labels] * 4 + rng.normal(scale=.05, size=(len(labels), 5))
            write_npz_atomic(dest, object_ids=np.asarray([r["object_id"] for r in rows]),
                             labels=labels, alerts=np.asarray([r["alert"] for r in rows], dtype=np.int64),
                             vit_cls=features, expert_concat=features + .05)
            write_json_atomic(dest.with_suffix(".json"), {
                "schema": "task5a_feature_shard_v1", "plan_sha256": plan["plan_sha256"],
                "sha256": digest_file(dest), "split": split, "shard": shard,
            })
    script.report(argparse.Namespace(run_dir=run_dir))
    report = json.loads((run_dir / "report.json").read_text())
    assert set(report["representations"]) == {"vit_cls", "expert_concat"}
    assert report["representations"]["vit_cls"]["splits"]["validation"]["macro_f1"] > .8
    script.report(argparse.Namespace(run_dir=run_dir))  # idempotent
    bad = run_dir / "features" / "test" / "shard_000_of_002.npz"
    bad.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Corrupt feature shard"):
        script.report(argparse.Namespace(run_dir=run_dir))
