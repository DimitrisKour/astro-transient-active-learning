#!/usr/bin/env python3
"""Read-only, CPU-only temperature calibration audit for Task4(.85) vs Task6A(.5).

Object-level cross-fitted calibration on *validation only*. Optionally assesses
entropy-ranking changes using *saved* next-cycle unlabeled-pool logits. No
new inference, no model import, no changes to experiments, and no test access.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

SEEDS = (135, 246, 357)
POLICIES = ("entropy", "entropy_balanced_explore")
CYCLES = (0, 3, 5, 9)
SUITES = {"0.85": "astrominn_closed_loop_warm_v1", "0.5": "astrominn_closed_loop_warm_weight05_newl0_v1"}
# Task 4 reused Experiment B L0 checkpoints instead of training its own.
BOOTSTRAP_SOURCE_SUITES = {"0.85": "astrominn_historical_replay_v1", "0.5": SUITES["0.5"]}
CLASS_NAMES = ("AGN", "TDE", "SN I", "SN II", "CV")
DEFAULT_PERSISTENT = Path("/projects/bcrv/dkourtesis/applecider_adaptive")
DEFAULT_DATA = Path("/work/hdd/bcrv/dkourtesis/applecider_adaptive")
DEFAULT_OUTPUT = DEFAULT_PERSISTENT / "diagnostics" / "temperature_05_vs_085_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and strictly positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    exps = np.exp(z)
    return exps / exps.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, targets: np.ndarray, temp: float) -> float:
    z = np.asarray(logits, dtype=np.float64) / temp
    maximum = np.max(z, axis=1)
    shifted = z - maximum[:, None]
    logsumexp = maximum + np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(logsumexp - z[np.arange(len(targets)), targets]))


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Bounded golden-section search over log T in [log(.05), log(20)]."""
    lo, hi = math.log(0.05), math.log(20.0)
    tau = (math.sqrt(5) - 1) / 2
    a, b = hi - tau*(hi-lo), lo + tau*(hi-lo)
    fa, fb = nll(logits, targets, math.exp(a)), nll(logits, targets, math.exp(b))
    for _ in range(115):
        if fa > fb:
            lo, a, fa = a, b, fb
            b = lo + tau*(hi-lo)
            fb = nll(logits, targets, math.exp(b))
        else:
            hi, b, fb = b, a, fa
            a = hi - tau*(hi-lo)
            fa = nll(logits, targets, math.exp(a))
    return math.exp((lo+hi)/2)


def deterministic_stratified_folds(ids: np.ndarray, targets: np.ndarray, k: int = 5, seed: int = 4310) -> np.ndarray:
    """Stable object-ID based stratification, identical across suites/models."""
    if len(ids) != len(targets) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Object IDs must be unique, aligned with labels")
    if np.any(np.bincount(targets, minlength=5) < k):
        raise ValueError(f"Each validation class needs >= {k} objects for stratified CV")
    folds = np.empty(len(ids), dtype=np.int64)
    for cls in range(5):
        which = np.where(targets == cls)[0].tolist()
        which.sort(key=lambda i: hashlib.sha256(f"{seed}|{ids[i]}".encode()).hexdigest())
        for j, idx in enumerate(which):
            folds[idx] = j % k
    return folds


def probability_metrics(p: np.ndarray, y: np.ndarray, bins: int = 15) -> dict[str,float]:
    if p.ndim != 2 or p.shape[1] != 5 or len(y) != len(p):
        raise ValueError("Invalid class probability shape")
    eps = 1e-300
    n = len(y)
    labels = np.eye(5, dtype=np.float64)[y]
    confidence = np.max(p, axis=1)
    correct = (np.argmax(p, axis=1) == y).astype(float)
    indices = np.minimum((confidence*bins).astype(int), bins-1)
    ece = 0.0
    for b in range(bins):
        selected = indices == b
        if selected.any():
            ece += float(selected.mean()*abs(correct[selected].mean()-confidence[selected].mean()))
    return {
        "nll": float(-np.mean(np.log(np.maximum(p[np.arange(n), y], eps)))),
        "brier": float(np.mean(np.sum((p-labels)**2, axis=1))),
        "ece_top_label_15bin": ece,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
    }


def per_class_metrics(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str,float|str]]:
    rows=[]
    for cls, name in enumerate(CLASS_NAMES):
        target=(y == cls).astype(float)
        conf=p[:,cls]
        indices=np.minimum((conf*bins).astype(int),bins-1)
        ece=0.0
        for b in range(bins):
            mask=indices == b
            if mask.any():
                ece += float(mask.mean()*abs(target[mask].mean()-conf[mask].mean()))
        rows.append({"class": name, "n_positive":int(target.sum()),
                     "one_vs_rest_brier":float(np.mean((conf-target)**2)),
                     "one_vs_rest_ece_10bin":ece,
                     "mean_probability":float(conf.mean()),
                     "prevalence":float(target.mean())})
    return rows


def load_validation(path: Path) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        required={"object_ids","object_targets","object_values"}
        if not required.issubset(f.files):
            raise ValueError(f"Not an AstroMiNN evaluation NPZ: {path}; {f.files}")
        ids=np.asarray(f["object_ids"],dtype=str)
        targets=np.asarray(f["object_targets"],dtype=np.int64)
        logits=np.asarray(f["object_values"],dtype=np.float64)
    if logits.shape != (len(ids),5) or targets.shape != (len(ids),):
        raise ValueError(f"Invalid shapes: {path}")
    if len(set(ids.tolist())) != len(ids) or np.any(targets<0) or np.any(targets>=5) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid evaluation contents: {path}")
    return ids,targets,logits


def checkpoints(data: Path, weight: str, seed: int, policy: str, cycle: int) -> tuple[Path,Path]:
    """Resolve checkpoint provenance independently from its evaluation suite.

    Task 4 (0.85) borrows Experiment B's L0 checkpoint, whereas Task 6A (0.5)
    trains its own L0. Cycle>=1 checkpoints are stored in their own suite.
    """
    suite=SUITES[weight]
    base=data/"runs"/suite
    if cycle == 0:
        checkpoint=(data/"runs"/BOOTSTRAP_SOURCE_SUITES[weight]/"shared"/
                    f"seed_{seed}"/"bootstrap_training"/"best_vit_tower.pth")
    else:
        checkpoint=base/policy/f"seed_{seed}"/f"cycle_{cycle:03d}"/"training"/"best_vit_tower.pth"
    evaluation=base/policy/f"seed_{seed}"/f"cycle_{cycle:03d}"/"evaluation"
    return checkpoint,evaluation


def expected_paths(persistent: Path, data: Path, weight: str, seed: int, policy: str, cycle: int) -> dict[str,Path]:
    suite=SUITES[weight]
    checkpoint,eval_dir=checkpoints(data,weight,seed,policy,cycle)
    status=persistent/"runs"/suite/policy/f"seed_{seed}"/"run_state.json"
    provenance=(data/"runs"/BOOTSTRAP_SOURCE_SUITES[weight]/"shared"/f"seed_{seed}"/"bootstrap_training"/"training_summary.json" if cycle==0
                else data/"runs"/suite/policy/f"seed_{seed}"/f"cycle_{cycle:03d}"/"training"/"training_summary.json")
    return {"checkpoint":checkpoint,"validation":eval_dir/"validation_predictions.npz", "evaluation":eval_dir/"evaluation_summary.json", "status":status,"training":provenance,
            "pool":data/"runs"/suite/policy/f"seed_{seed}"/f"cycle_{cycle+1:03d}"/"inference.npz",
            "pool_summary":persistent/"runs"/suite/policy/f"seed_{seed}"/f"cycle_{cycle+1:03d}"/"cycle_summary.json"}


def audit_provenance(paths: dict[str,Path], weight: str, cycle: int) -> None:
    train=read_json(paths["training"])
    actual=float(train["policy"]["class_weight_exponent"])
    if abs(actual-float(weight))>1e-8:
        raise ValueError(f"Weight exponent provenance mismatch {actual} != {weight}: {paths['training']}")
    if train["policy"].get("new_object_weighting")!="per_sample":
        raise ValueError("Expected per_sample objective")
    evaluation=read_json(paths["evaluation"])
    if tuple(evaluation.get("model_class_names",[])) != CLASS_NAMES:
        raise ValueError("Class order mismatch in evaluation JSON")
    if Path(evaluation["checkpoint_path"]) != paths["checkpoint"]:
        raise ValueError("Evaluation was not made from the expected checkpoint")
    split_names=[s["split"] for s in evaluation["splits"]]
    if "validation" not in split_names:
        raise ValueError("Validation missing in evaluation summary")
    if not paths["checkpoint"].is_file():
        raise FileNotFoundError(paths["checkpoint"])
    if cycle>0:
        status=read_json(paths["status"])
        if int(status.get("last_completed_cycle",-1)) < cycle:
            raise RuntimeError(f"Cycle {cycle} incomplete")


def crossfit(logits: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray,np.ndarray,list[float]]:
    original=softmax(logits,1.0)
    corrected=np.zeros_like(original)
    temps=[]
    for fold in range(5):
        test=folds == fold
        train=~test
        temp=fit_temperature(logits[train],y[train])
        corrected[test]=softmax(logits[test],temp)
        temps.append(temp)
    if not np.allclose(corrected.sum(axis=1),1.0):
        raise RuntimeError("Invalid OOF probabilities")
    if not np.array_equal(original.argmax(axis=1), corrected.argmax(axis=1)):
        raise RuntimeError("Temperature scaling changed predicted class")
    return original,corrected,temps


def pool_effect(path: Path, summary_path: Path, checkpoint: Path, temperature: float, k: int=473) -> dict[str,Any]:
    if not path.is_file():
        return {"pool_status":"missing_saved_inference","pool_n":None,"topk_overlap":None}
    if not summary_path.is_file():
        return {"pool_status":"missing_next_cycle_summary","pool_n":None,"topk_overlap":None}
    summary=read_json(summary_path)
    if Path(summary.get("inference_checkpoint","")) != checkpoint:
        raise ValueError(f"Pool logits checkpoint lineage mismatch: {summary_path}")
    with np.load(path,allow_pickle=False) as f:
        if str(f["output_type"].item()) != "logits":
            raise ValueError(f"Cannot temperature-scale non-logit pool outputs: {path}")
        ids=np.asarray(f["object_ids"],dtype=str)
        logits=np.asarray(f["values"],dtype=np.float64)
    if logits.shape != (len(ids),5) or len(set(ids.tolist())) != len(ids) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid pool inference: {path}")
    k=min(k,len(ids))
    pa,pb=softmax(logits,1.0),softmax(logits,temperature)
    def entropy(p):return -np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)
    e1,e2=entropy(pa),entropy(pb)
    def top(score):
        order=np.lexsort((ids,-score))
        return set(ids[order[:k]].tolist())
    a,b=top(e1),top(e2)
    return {"pool_status":"ok", "pool_n":len(ids),"topk_n":k,"topk_overlap":len(a & b),
            "topk_overlap_fraction":len(a & b)/k,"topk_exchanges":k-len(a & b),
            "pool_predicted_labels_unchanged":bool(np.array_equal(pa.argmax(axis=1),pb.argmax(axis=1)))}


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    names=list(dict.fromkeys(key for row in rows for key in row))
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=names);w.writeheader();w.writerows(rows)
    temp.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    temp.replace(path)


def run(persistent: Path, data: Path, output: Path, *, allow_incomplete: bool=False) -> dict:
    rows=[]; details=[]; paired=[]; pools=[]; missing=[]
    cache={}
    fixed_id_targets=None
    for weight,suite in SUITES.items():
        for seed in SEEDS:
            for policy in POLICIES:
                for cycle in CYCLES:
                    key=(weight,seed,policy,cycle)
                    paths=expected_paths(persistent,data,weight,seed,policy,cycle)
                    mandatory=("validation","evaluation","training","checkpoint")
                    absent=[str(paths[x]) for x in mandatory if not paths[x].is_file()]
                    if cycle>0 and not paths["status"].is_file():absent.append(str(paths["status"]))
                    if absent:
                        missing.append({"weight":weight,"seed":seed,"policy":policy,"cycle":cycle,"missing":absent})
                        continue
                    audit_provenance(paths,weight,cycle)
                    ids,y,z=load_validation(paths["validation"])
                    # Align and preserve fold assignment independent of model ordering.
                    order=np.argsort(ids)
                    ids,y,z=ids[order],y[order],z[order]
                    if fixed_id_targets is None:
                        fixed_id_targets=(ids,y)
                    elif not np.array_equal(fixed_id_targets[0],ids) or not np.array_equal(fixed_id_targets[1],y):
                        raise ValueError("Validation object IDs/targets differ between experiment configurations")
                    cache[key]=(ids,y,z,paths)
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"{len(missing)} requested model records missing; see --allow-incomplete or wait for Task6A to finish. First: {missing[0]}")
    if not cache:
        raise FileNotFoundError("No validation artifacts are ready")
    ids,y=fixed_id_targets
    folds=deterministic_stratified_folds(ids,y)
    # Cache cycle-zero calibration for each weight/seed: L0 is shared by both policies.
    results={}
    for key,(ids,y,z,paths) in cache.items():
        weight,seed,policy,cycle=key
        common_key=(weight,seed,"bootstrap",0) if cycle==0 else key
        if common_key in results:
            result=results[common_key]
        else:
            baseline,calibrated,temperatures=crossfit(z,y,folds)
            final_t=fit_temperature(z,y)
            ref=probability_metrics(baseline,y); scaled=probability_metrics(calibrated,y)
            result={"raw":ref,"scaled":scaled,"fold_t":temperatures,"final_t":final_t,
                    "per_class_raw":per_class_metrics(baseline,y),"per_class_scaled":per_class_metrics(calibrated,y),
                    "original_logits":z.copy() if cycle==0 else None}
            results[common_key]=result
        if cycle==0 and result["original_logits"] is not None and not np.allclose(result["original_logits"],z,rtol=1e-6,atol=1e-8):
            raise ValueError(f"L0 predictions differ by policy for {weight} seed {seed}")
        raw,scaled=result["raw"],result["scaled"]
        row={"weight":weight,"suite":SUITES[weight],"seed":seed,"policy":policy,"cycle":cycle,"n_validation_objects":len(y),
             "fitted_temperature_all_validation":result["final_t"],
             "crossfit_t_mean":float(np.mean(result["fold_t"])),"crossfit_t_min":float(min(result["fold_t"])),
             "crossfit_t_max":float(max(result["fold_t"])), "model_predicted_label_change_count":0}
        for m in raw:
            row[f"raw_{m}"]=raw[m];row[f"crossfit_{m}"]=scaled[m];row[f"delta_{m}"]=scaled[m]-raw[m]
        rows.append(row)
        for cls,ra,ca in zip(CLASS_NAMES,result["per_class_raw"],result["per_class_scaled"],strict=True):
            details.append({"weight":weight,"seed":seed,"policy":policy,"cycle":cycle,"class":cls,
                            "n_positive":ra["n_positive"],
                            "raw_brier":ra["one_vs_rest_brier"],"scaled_brier":ca["one_vs_rest_brier"],
                            "raw_ece":ra["one_vs_rest_ece_10bin"],"scaled_ece":ca["one_vs_rest_ece_10bin"],
                            "delta_brier":ca["one_vs_rest_brier"]-ra["one_vs_rest_brier"],
                            "delta_ece":ca["one_vs_rest_ece_10bin"]-ra["one_vs_rest_ece_10bin"]})
        if cycle==9:
            pool={"pool_status":"pool_exhausted","pool_n":0,"topk_overlap":None}
        else:
            pool=pool_effect(paths["pool"],paths["pool_summary"],paths["checkpoint"],result["final_t"])
        pools.append({"weight":weight,"seed":seed,"policy":policy,"cycle":cycle,
                      "source_next_cycle":None if cycle==9 else cycle+1,
                      "interpretation":"entropy_topk_proxy_only_not_balanced_exploration_policy" if policy!="entropy" else "entropy_policy_topk",
                      **pool})
    lookup={(r["weight"],r["seed"],r["policy"],r["cycle"]):r for r in rows}
    for seed in SEEDS:
        for policy in POLICIES:
            for cycle in CYCLES:
                a,b=lookup.get(("0.5",seed,policy,cycle)),lookup.get(("0.85",seed,policy,cycle))
                if not a or not b: continue
                p={"seed":seed,"policy":policy,"cycle":cycle,"n_validation_objects":a["n_validation_objects"],
                   "temp05":a["fitted_temperature_all_validation"],"temp085":b["fitted_temperature_all_validation"]}
                for metric in ("nll","brier","ece_top_label_15bin","accuracy"):
                    for prefix in ("raw","crossfit"):
                        p[f"{prefix}_05_{metric}"]=a[f"{prefix}_{metric}"]
                        p[f"{prefix}_085_{metric}"]=b[f"{prefix}_{metric}"]
                        p[f"{prefix}_05_minus_085_{metric}"]=a[f"{prefix}_{metric}"]-b[f"{prefix}_{metric}"]
                paired.append(p)
    summary={"schema":"task6a_temperature_audit_v1","weights_to_suites":SUITES,"bootstrap_source_suites":BOOTSTRAP_SOURCE_SUITES,"seeds":SEEDS,"policies":POLICIES,
             "cycles":CYCLES,"expected_rows":48,"actual_rows":len(rows),"paired_rows":len(paired),
             "missing":missing,"calibration":"object-level; five-fold deterministic stratified cross-fitting; NLL-optimal scalar T fitted on other folds; separate final T on all validation for prospective pool ranking",
             "validation_caveat":"Validation also selected model checkpoints; cross-fitting prevents temperature-fit contamination of audit folds but cannot undo checkpoint selection on validation.",
             "pool_caveat":"Saved next-cycle inference; entropy top-k is an exact entropy-policy proxy, NOT the full balanced-exploration acquisition; cycle9 has no U remaining.",
             "test":"Never accessed", "source":"existing NPZ logits and training/evaluation metadata only; no GPU inference"}
    atomic_csv(output/"calibration_per_model.csv",rows)
    atomic_csv(output/"paired_05_vs_085.csv",paired)
    atomic_csv(output/"per_class_calibration.csv",details)
    atomic_csv(output/"pool_entropy_overlap.csv",pools)
    atomic_json(output/"summary.json",summary)
    print(json.dumps({"output":str(output),"models":len(rows),"pairs":len(paired),"missing":len(missing),"pool_status":{x:sum(r["pool_status"]==x for r in pools) for x in set(r["pool_status"] for r in pools)}},indent=2))
    return summary


def read_dependency_jobs(manifest: Path) -> list[str]:
    data=read_json(manifest)
    if not isinstance(data,dict) or not isinstance(data.get("jobs"),list):
        raise ValueError("Task6A submission manifest requires a jobs list")
    jobs=data["jobs"]
    if len(jobs)!=57:
        raise ValueError(f"Expected all 57 Task6A jobs; got {len(jobs)}")
    identifiers={(j["seed"],j["policy"],j["cycle"]):j["job_id"] for j in jobs}
    if len(identifiers)!=57 or len(set(identifiers.values()))!=57:
        raise ValueError("Duplicate Task6A submitted jobs or IDs")
    for seed in SEEDS:
        if (seed,"bootstrap",0) not in identifiers:raise ValueError(f"Missing L0 job for seed {seed}")
        for policy in POLICIES:
            for cycle in range(1,10):
                key=(seed,policy,cycle)
                if key not in identifiers:raise ValueError(f"Missing {key}")
                parent=identifiers[(seed,"bootstrap",0)] if cycle==1 else identifiers[(seed,policy,cycle-1)]
                record=next(j for j in jobs if (j["seed"],j["policy"],j["cycle"])==key)
                if str(record.get("dependency_job_id"))!=str(parent):
                    raise ValueError(f"Wrong dependency for {key}")
    return [str(identifiers[(seed,policy,9)]) for seed in SEEDS for policy in POLICIES]


def schedule_cpu(manifest: Path, script: Path, sbatch_file: Path, output: Path, *, do_submit: bool=False) -> None:
    ids=read_dependency_jobs(manifest)
    marker=output/"cpu_job_submission.json"
    if marker.exists():raise FileExistsError(f"Already queued calibration audit: {marker}")
    cmd=["sbatch","--parsable", "--dependency=afterok:"+":".join(ids),
         "--output="+str(output/"temperature_%j.out"), "--error="+str(output/"temperature_%j.err"),
         str(sbatch_file),str(script),str(output)]
    print("SBATCH:", " ".join(cmd))
    if not do_submit:
        print("DRY RUN: nothing queued. Add --submit to queue one dependent CPU job.")
        return
    output.mkdir(parents=True,exist_ok=True)
    proc=subprocess.run(cmd,text=True,capture_output=True,check=True)
    job_id=proc.stdout.strip().split(";")[0]
    if not job_id.isdigit():raise RuntimeError(f"Unexpected sbatch output {proc.stdout!r}")
    atomic_json(marker,{"job_id":job_id,"dependency_final_cycles":ids,"command":cmd})
    print(f"Submitted CPU calibration job {job_id}, dependent on all six 6A final cycles")


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("plan","run","submit"))
    ap.add_argument("--persistent",type=Path,default=DEFAULT_PERSISTENT)
    ap.add_argument("--data",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--manifest",type=Path,default=DEFAULT_PERSISTENT/"closed_loop"/SUITES["0.5"]/"submitted.json")
    ap.add_argument("--allow-incomplete",action="store_true")
    ap.add_argument("--submit",action="store_true",help="Only valid with submit action; actually queue dependent CPU job")
    a=ap.parse_args()
    if a.action=="plan":
        for w in SUITES:
            for s in SEEDS:
                for p in POLICIES:
                    for c in CYCLES:
                        paths=expected_paths(a.persistent,a.data,w,s,p,c)
                        print(f"weight={w} seed={s} policy={p} cycle={c} val={paths['validation']}")
        print("48 comparison rows (42 unique checkpoints; cycle0 repeated across policies), no GPU or model inference")
    elif a.action=="run":
        run(a.persistent,a.data,a.output,allow_incomplete=a.allow_incomplete)
    elif a.action=="submit":
        if not a.manifest.is_file():raise FileNotFoundError(a.manifest)
        schedule_cpu(a.manifest,Path(__file__).resolve(),Path(__file__).with_name("task6a_temperature_audit.sbatch"),a.output,do_submit=a.submit)

if __name__=="__main__":main()
