from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/"scripts/adaptive/task6a_temperature_audit.py"
spec=importlib.util.spec_from_file_location("task6a_temperature_audit",SCRIPT)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
m=module


def test_scalar_temperature_softmax_and_invariance():
    z=np.array([[4,1,0,-1,-2],[1,2,1,-1,0]],float)
    p=m.softmax(z,1)
    assert np.allclose(p.sum(axis=1),1)
    assert np.array_equal(p.argmax(axis=1),m.softmax(z,6).argmax(axis=1))
    assert m.softmax(z,3)[0].max() < p[0].max()


def test_fit_temperature_softens_overconfident_probabilities():
    rng=np.random.default_rng(2026)
    y=np.arange(5).repeat(45)
    z=rng.normal(0,0.9,size=(len(y),5))
    z[np.arange(len(y)),y]+=0.8
    z*=7
    t=m.fit_temperature(z,y)
    assert t > 1.05
    assert m.nll(z,y,t) < m.nll(z,y,1.0)


def test_stratification_invariant_to_order():
    ids=np.asarray([f"object-{i:03d}" for i in range(75)])
    y=np.arange(75)%5
    fold=m.deterministic_stratified_folds(ids,y)
    perm=np.random.default_rng(6).permutation(75)
    fold2=m.deterministic_stratified_folds(ids[perm],y[perm])
    assert np.array_equal(fold,fold2[np.argsort(perm)])
    assert set(fold)==set(range(5))
    for c in range(5):
        assert len(set(fold[y==c]))==5


def test_probability_metrics_exact_probabilities():
    y=np.arange(5)
    p=np.eye(5)[y]
    metrics=m.probability_metrics(p,y)
    assert metrics["nll"]==0
    assert metrics["brier"]==0
    assert metrics["ece_top_label_15bin"]==0
    assert metrics["accuracy"]==1


def make_manifest(path):
    rows=[]; ids={}
    for seed in m.SEEDS:
        ids[(seed,"bootstrap",0)]=str(10_000+seed)
        rows.append({"seed":seed,"cycle":0,"policy":"bootstrap","job_id":ids[(seed,"bootstrap",0)],"dependency_job_id":None})
    n=40000
    for seed in m.SEEDS:
        for policy in m.POLICIES:
            for c in range(1,10):
                job=str(n);n+=1
                parent=ids[(seed,"bootstrap",0)] if c==1 else ids[(seed,policy,c-1)]
                rows.append({"seed":seed,"policy":policy,"cycle":c,"job_id":job,"dependency_job_id":parent})
                ids[(seed,policy,c)]=job
    path.write_text(json.dumps({"status":"held","jobs":rows}))
    return rows


def test_dependency_manifest_and_failure(tmp_path):
    path=tmp_path/"submitted.json"
    rows=make_manifest(path)
    finals=m.read_dependency_jobs(path)
    assert len(finals)==6
    assert len(set(finals))==6
    rows[-1]["dependency_job_id"]="wrong"
    path.write_text(json.dumps({"jobs":rows}))
    with pytest.raises(ValueError,match="Wrong dependency"):
        m.read_dependency_jobs(path)


def write_mock_model(persistent,data,weight,seed,policy,cycle,ids,y,logits):
    suite=m.SUITES[weight]
    paths=m.expected_paths(persistent,data,weight,seed,policy,cycle)
    for k in ("checkpoint","validation","evaluation","training","status"):
        paths[k].parent.mkdir(parents=True,exist_ok=True)
    paths["checkpoint"].write_bytes(b"placeholder model")
    with paths["validation"].open("wb") as f:
        np.savez_compressed(f,object_ids=ids,object_targets=y,object_values=logits)
    paths["training"].write_text(json.dumps({"policy":{"class_weight_exponent":float(weight),"new_object_weighting":"per_sample"}}))
    paths["evaluation"].write_text(json.dumps({"checkpoint_path":str(paths["checkpoint"]),"model_class_names":list(m.CLASS_NAMES),"splits":[{"split":"validation"}]}))
    paths["status"].write_text(json.dumps({"last_completed_cycle":9}))
    if cycle<9:
        paths["pool"].parent.mkdir(parents=True,exist_ok=True)
        with paths["pool"].open("wb") as f:
            np.savez_compressed(f,object_ids=np.array(["p0","p1","p2"]),values=logits[:3],output_type="logits")
        paths["pool_summary"].parent.mkdir(parents=True,exist_ok=True)
        paths["pool_summary"].write_text(json.dumps({"inference_checkpoint":str(paths["checkpoint"])}))


def test_full_48_rows_read_only_report(tmp_path):
    persistent=tmp_path/"p";data=tmp_path/"d";out=tmp_path/"out"
    ids=np.asarray([f"obj{i:03d}" for i in range(100)])
    y=np.arange(100)%5
    rng=np.random.default_rng(17)
    for weight in m.SUITES:
        for seed in m.SEEDS:
            for policy in m.POLICIES:
                for cycle in m.CYCLES:
                    z=rng.normal(size=(100,5)) + (np.eye(5)[y]*1.2)
                    if cycle == 0:
                        # Baseline logits must agree for the two policies.
                        z=np.random.default_rng(seed+int(float(weight)*100)).normal(size=(100,5)) + (np.eye(5)[y]*1.2)
                    write_mock_model(persistent,data,weight,seed,policy,cycle,ids,y,z)
    result=m.run(persistent,data,out)
    assert result["actual_rows"]==48
    assert result["paired_rows"]==24
    assert len(list(__import__("csv").DictReader((out/"pool_entropy_overlap.csv").open())))==48
    assert len(list(__import__("csv").DictReader((out/"per_class_calibration.csv").open())))==240
    assert not list(tmp_path.rglob("test_predictions.npz"))
    # Saved inputs are unchanged and no model inference took place.
    assert (out/"summary.json").is_file()


def test_no_missing_data_without_allow_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError,match="requested model records missing"):
        m.run(tmp_path/"p",tmp_path/"d",tmp_path/"o")


def test_cycle_zero_checkpoint_source_is_original_experiment_b(tmp_path):
    p,d=tmp_path/"p",tmp_path/"d"
    for seed in m.SEEDS:
        for policy in m.POLICIES:
            old=m.expected_paths(p,d,"0.85",seed,policy,0)
            new=m.expected_paths(p,d,"0.5",seed,policy,0)
            assert "astrominn_historical_replay_v1/shared" in str(old["checkpoint"])
            assert "astrominn_historical_replay_v1/shared" in str(old["training"])
            assert "astrominn_closed_loop_warm_v1" in str(old["validation"])
            assert "astrominn_closed_loop_warm_weight05_newl0_v1/shared" in str(new["checkpoint"])
            assert "astrominn_closed_loop_warm_weight05_newl0_v1/shared" in str(new["training"])
            for weight in m.SUITES:
                later=m.expected_paths(p,d,weight,seed,policy,3)
                assert m.SUITES[weight] in str(later["training"])
                assert m.SUITES[weight] in str(later["checkpoint"])
