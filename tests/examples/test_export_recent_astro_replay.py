"""Strict, CPU-only tests for recent AL trajectory export."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

FILE = Path(__file__).resolve().parents[2] / "scripts/examples/export_recent_astro_replay.py"
spec = importlib.util.spec_from_file_location("recent_school_export", FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixtures(tmp_path, *, strategy="entropy", seed=135, metrics=True):
    run = tmp_path / "runs" / "suite" / strategy / f"seed_{seed}"
    run.mkdir(parents=True)
    (run / "run_state.json").write_text(json.dumps({"last_completed_cycle": 9, "active_cycle": None}))
    hist = [{"cycle": c, "split": "validation", "strategy": strategy, "seed": seed,
             "n_labeled": 20 + c*5, "n_acquired": 0 if c == 0 else 5,
             "metrics": {"f1_macro": .3+c*.01, "balanced_accuracy": .4+c*.01} if metrics else {"f1_macro": .3}}
            for c in range(10)]
    hist.append({"cycle": 9, "split": "test", "n_labeled": 65, "metrics": {"f1_macro": .99}})
    (run / "evaluation_history.json").write_text(json.dumps(hist))
    return run


def test_collect_validation_only(tmp_path):
    fixtures(tmp_path)
    rows, prov = mod.collect(tmp_path, {"suite": ("entropy",)}, (135,))
    assert len(rows) == 20
    assert {r["split"] for r in rows} == {"validation"}
    assert rows[-1]["n_acquired_total"] == 45
    assert prov["run_count"] == 1
    assert prov["baseline_audit"]["135"]["baseline_spread"] == 0
    assert ".99" not in mod.csv_text(rows, mod.CSV_FIELDS)


def test_incomplete_and_missing_metrics_fail(tmp_path):
    run = fixtures(tmp_path)
    state_file = run / "run_state.json"
    state_file.write_text(json.dumps({"last_completed_cycle": 8, "active_cycle": None}))
    with pytest.raises(ValueError, match="not completed"):
        mod.collect(tmp_path, {"suite": ("entropy",)}, (135,))
    state_file.write_text(json.dumps({"last_completed_cycle": 9, "active_cycle": None}))
    history = json.loads((run / "evaluation_history.json").read_text())
    history[3]["metrics"].pop("balanced_accuracy")
    (run / "evaluation_history.json").write_text(json.dumps(history))
    with pytest.raises(ValueError, match="missing balanced_accuracy"):
        mod.collect(tmp_path, {"suite": ("entropy",)}, (135,))


def test_nonmonotonic_and_duplicate_fail(tmp_path):
    run = fixtures(tmp_path)
    hp = run / "evaluation_history.json"
    hist = json.loads(hp.read_text())
    hist[4]["n_labeled"] = 1
    hp.write_text(json.dumps(hist))
    with pytest.raises(ValueError, match="nonmonotonic"):
        mod.collect(tmp_path, {"suite": ("entropy",)}, (135,))
    hist[4]["n_labeled"] = 40
    hist.append(dict(hist[1]))
    hp.write_text(json.dumps(hist))
    with pytest.raises(ValueError, match="duplicate validation"):
        mod.collect(tmp_path, {"suite": ("entropy",)}, (135,))