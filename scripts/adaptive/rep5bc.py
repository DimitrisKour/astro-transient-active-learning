#!/usr/bin/env python3
"""Prepare, validate, run or explicitly submit isolated representation experiments.

5B: fixed-acquisition matched-manifest ViT-full/head-only fits.
5C: independent closed-loop ViT-full/head-only acquisition trajectories.
No submission without an explicit --submit, and no mutation of reference runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEEDS = (135, 246, 357)
POLICIES = ("entropy", "entropy_balanced_explore")
MODES = ("vit_full", "vit_head_only")
BASE_SUITE = "astrominn_historical_replay_v1"
PROJECT = Path("/projects/bcrv/dkourtesis/applecider_adaptive")
DATA = Path("/work/hdd/bcrv/dkourtesis/applecider_adaptive")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def dump(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise ValueError(f"Immutable plan or report differs: {path}")
        return
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def checked_names(items, allowed, label):
    values = tuple(items)
    if not values or len(values) != len(set(values)) or set(values) - set(allowed):
        raise ValueError(f"Invalid {label} list: {values}; allowed {allowed}")
    return values


def patched_training_source(content: bytes) -> bytes:
    """Opt-in, guarded 5B/C training fix applied ONLY to frozen source copy."""
    source = content.decode("utf-8")
    old = ("    for parameter in model.vit_tower.parameters():\n"
           "        parameter.requires_grad = True\n")
    new = ("    head_only = getattr(model, '_applecider_vit_head_only', False)\n"
           "    for parameter in model.vit_tower.parameters():\n"
           "        parameter.requires_grad = not head_only\n"
           "    if head_only:\n"
           "        for parameter in model.vit_tower.fc_out.parameters():\n"
           "            parameter.requires_grad = True\n")
    if source.count(old) != 1:
        raise ValueError("Upstream ViT freeze routine changed; refusing unverified 5B/C adaptation")
    source = source.replace(old, new, 1)
    old = "    model.vit_tower.train()\n"
    new = ("    model.vit_tower.train()\n"
           "    if getattr(model, '_applecider_vit_head_only', False):\n"
           "        model.vit_tower.eval()\n"
           "        model.vit_tower.fc_out.train()\n")
    if source.count(old) != 1:
        raise ValueError("Upstream training-mode routine changed; refusing unverified 5B/C adaptation")
    source = source.replace(old, new, 1)
    compile(source, "5bc_frozen_training.py", "exec")
    return source.encode("utf-8")


def snapshot_files(repo: Path) -> list[Path]:
    required = [
        *sorted((repo / "src/al_pipeline").rglob("*.py")),
        repo / "scripts/adaptive/rep5bc.py",
        repo / "scripts/adaptive/rep5bc_worker.sbatch",
        repo / "scripts/adaptive/rep5bc_build_runner.py",
        repo / "scripts/run_representation_5c.py",
    ]
    if not required or any(not item.is_file() for item in required):
        raise FileNotFoundError("5B/5C package/source incomplete")
    return required


def freeze(repo: Path, work: Path) -> dict:
    """Freeze sources with head-only fix ONLY within the new suite."""
    files = snapshot_files(repo)
    payloads = {}
    for item in files:
        name = str(item.relative_to(repo))
        content = item.read_bytes()
        if name == "src/al_pipeline/astronomy/astrominn/training.py":
            content = patched_training_source(content)
        payloads[name] = content
    desired = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    manifest = work / "source_snapshot.json"
    root = work / "source"
    if manifest.exists():
        if json.loads(manifest.read_text()) != desired:
            raise ValueError("5B/5C sources changed after freeze; use a new work root")
        verify_source(work)
        return desired
    if root.exists():
        raise FileExistsError(f"Source exists without completed freeze manifest: {root}")
    root.mkdir(parents=True)
    for name, content in payloads.items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    dump(manifest, desired)
    verify_source(work)
    return desired


def verify_source(work: Path) -> None:
    manifest = work / "source_snapshot.json"
    checks = json.loads(manifest.read_text())
    for name, digest in checks.items():
        dest = work / "source" / name
        if not dest.is_file() or sha256(dest) != digest:
            raise ValueError(f"Frozen source missing/changed: {dest}")


def load_plan(work: Path, kind: str) -> dict:
    path = work / "plan.json"
    item = json.loads(path.read_text())
    if item.get("schema") != f"representation_{kind}_v1":
        raise ValueError(f"Incorrect {kind} work root: {work}")
    verify_source(work)
    if sha256(Path(item["registry"])) != item["registry_sha256"]:
        raise ValueError("Registry changed after plan")
    if kind == "5c" and sha256(Path(item["excluded_manifest"])) != item["excluded_manifest_sha256"]:
        raise ValueError("5C exclusion audit source has changed")
    return item


def prepare_b(args) -> None:
    """Pre-register paired fits from immutable source cumulative-manifest paths."""
    policies = checked_names(args.policies, POLICIES, "policies")
    seeds = checked_names(args.seeds, SEEDS, "seeds")
    cycles = tuple(args.cycles)
    if not cycles or len(set(cycles)) != len(cycles) or min(cycles) < 1 or max(cycles) > 9:
        raise ValueError("5B must use unique source cycles in 1..9")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Positive training budget required")
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    resolved = DatasetRegistry.from_toml(args.registry).resolve_profile("astrominn_ztf")
    dataset = resolved.dataset
    if not dataset.validation_manifest or not Path(dataset.validation_manifest).is_file():
        raise FileNotFoundError("Fixed validation manifest missing")
    expert = resolved.profile.options.get("checkpoint")
    if not expert or not Path(expert).is_file():
        raise FileNotFoundError("Published expert checkpoint missing")
    original_l0 = args.persistent_root / "runs" / BASE_SUITE / "shared/historical_replay_split/initial_labeled_manifest.csv"
    if not original_l0.is_file():
        raise FileNotFoundError(f"Missing immutable original L0: {original_l0}")
    tasks = []
    for policy in policies:
        for seed in seeds:
            bootstrap = args.data_root / "runs" / BASE_SUITE / "shared" / f"seed_{seed}" / "bootstrap_training/best_vit_tower.pth"
            if not bootstrap.is_file():
                raise FileNotFoundError(f"Missing original L0 for seed {seed}: {bootstrap}")
            for cycle in cycles:
                summary = (args.persistent_root / "runs" / args.source_suite / policy
                           / f"seed_{seed}" / f"cycle_{cycle:03d}" / "cycle_summary.json")
                if not summary.is_file():
                    raise FileNotFoundError(f"Source cycle not completed: {summary}")
                cycle_info = json.loads(summary.read_text())
                if int(cycle_info.get("cycle", -1)) != cycle or cycle_info.get("strategy") != policy:
                    raise ValueError(f"Incompatible source cycle: {summary}")
                manifest = Path(cycle_info["cumulative_manifest"])
                if not manifest.is_file():
                    raise FileNotFoundError(manifest)
                for mode in MODES:
                    tasks.append({
                        "source_cycle_summary": str(summary), "source_summary_sha256": sha256(summary),
                        "train_manifest": str(manifest), "train_sha256": sha256(manifest),
                        "initial_checkpoint": str(bootstrap), "checkpoint_sha256": sha256(bootstrap),
                        "cycle": cycle, "seed": seed, "policy": policy, "mode": mode,
                    })
    plan = {
        "schema": "representation_5b_v1", "source_suite": args.source_suite,
        "registry": str(args.registry.resolve()), "registry_sha256": sha256(args.registry),
        "artifact_dir": str(Path(expert).parent), "validation_manifest": str(dataset.validation_manifest),
        "validation_sha256": sha256(Path(dataset.validation_manifest)),
        "original_l0": str(original_l0), "original_l0_sha256": sha256(original_l0),
        "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "tasks": tasks,
        "protocol": "fixed source cumulative L_c; fixed original L0 initialization per fit; frozen pretrained experts; paired seed and manifest; validation-only model comparison; no acquisition",
    }
    generated = args.repo / "scripts/run_representation_5c.py"
    if not generated.is_file():
        raise FileNotFoundError("Generate 5C runner before freezing (rep5bc_build_runner.py)")
    freeze(args.repo, args.work_root)
    dump(args.work_root / "plan.json", plan)
    print(f"5B prepared: {len(tasks)} jobs, {len(tasks)//2} paired comparisons. No jobs queued.")
    print(f"Work root: {args.work_root}")


def verify_b_task(plan: dict, task: dict) -> None:
    for path, digest in (("train_manifest", "train_sha256"),
                         ("initial_checkpoint", "checkpoint_sha256"),
                         ("source_cycle_summary", "source_summary_sha256")):
        if sha256(Path(task[path])) != task[digest]:
            raise ValueError(f"5B source changed: {task[path]}")
    if sha256(Path(plan["validation_manifest"])) != plan["validation_sha256"]:
        raise ValueError("Fixed validation source changed")
    if sha256(Path(plan["original_l0"])) != plan["original_l0_sha256"]:
        raise ValueError("Fixed original L0 source changed")


def run_b(args) -> None:
    """One fully reproducible training fit, never acquisition or source modifications."""
    work = args.work_root
    plan = load_plan(work, "5b")
    if not 0 <= args.task_index < len(plan["tasks"]):
        raise IndexError(args.task_index)
    task = plan["tasks"][args.task_index]
    verify_b_task(plan, task)
    import torch
    from dataclasses import replace
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer, AstroMiNNViTTrainingConfig
    from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
    from al_pipeline.astronomy.representation.vi_tuning import make_vit_factory
    from al_pipeline.core.training.base import TrainingRequest
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import DEFAULT_LEGACY_CODE_DIR
    profile = DatasetRegistry.from_toml(plan["registry"]).resolve_profile("astrominn_ztf")
    dataset = replace(profile.dataset, train_manifest=Path(plan["original_l0"]))
    # Paired modes share all fit data, seed, initialization, optimizer, and weighting.
    policy = AstroMiNNViTTrainingConfig(
        batch_size=plan["batch_size"], validation_batch_size=2048,
        max_epochs=plan["epochs"], patience=plan["patience"],
        learning_rate=plan["learning_rate"], loader_seed=task["seed"],
        train_num_workers=0, validation_num_workers=0, class_weight_source="training_manifest",
        augment_training=True, augment_validation=False, new_object_weighting="per_sample",
    )
    output = (work / "fits" / task["policy"] / f"seed_{task['seed']}"
              / f"cycle_{task['cycle']:03d}" / task["mode"])
    device = torch.device(args.device)
    trainer = AstroMiNNViTTrainer(
        dataset=dataset, policy=policy, device=device, initialization="pretrained",
        artifact_dir=plan["artifact_dir"], model_factory=make_vit_factory(
            task["mode"], artifact_dir=plan["artifact_dir"], legacy_code_dir=DEFAULT_LEGACY_CODE_DIR),
    )
    request = TrainingRequest(
        train_manifest=Path(task["train_manifest"]), validation_manifest=Path(plan["validation_manifest"]),
        output_dir=output / "training", run_id=f"rep5b_{plan['source_suite']}_{task['policy']}_{task['seed']}_{task['mode']}",
        cycle=task["cycle"], strategy=task["policy"], seed=task["seed"],
        initial_checkpoint=Path(task["initial_checkpoint"]),
    )
    prior = trainer.completed_result(request)
    training = prior if prior is not None else trainer.train(request)
    evaluator = AstroMiNNViTEvaluator(
        dataset=dataset, device=device, artifact_dir=plan["artifact_dir"],
        manifests={"validation": plan["validation_manifest"]},
    )
    result = evaluator.evaluate(checkpoint_path=training.checkpoint_path, output_dir=output / "evaluation")
    evidence = {
        "schema": "representation_5b_fit_v1", "task_index": args.task_index, "task": task,
        "training_summary": str(training.summary_path), "evaluation_summary": str(result.summary_path),
    }
    dump(output / "fit_complete.json", evidence)
    print(json.dumps({"task_index": args.task_index, "output": str(output), "reused": prior is not None}))


def report_b(args) -> None:
    plan = load_plan(args.work_root, "5b")
    rows = []
    for index, task in enumerate(plan["tasks"]):
        dest = args.work_root / "fits" / task["policy"] / f"seed_{task['seed']}" / f"cycle_{task['cycle']:03d}" / task["mode"]
        info = dest / "fit_complete.json"
        if not info.exists():
            rows.append({"index": index, "mode": task["mode"], "seed": task["seed"], "policy": task["policy"],
                         "cycle": task["cycle"], "status": "missing"})
            continue
        completed = json.loads(info.read_text())
        if completed.get("task") != task:
            raise ValueError(f"5B completion identity differs: {info}")
        evaluation = json.loads((dest / "evaluation/evaluation_summary.json").read_text())
        summary = next(s for s in evaluation["splits"] if s["split"] == "validation")
        metrics = summary["object_metrics"]
        training = json.loads((dest / "training/training_summary.json").read_text())
        rows.append({"index": index, "mode": task["mode"], "seed": task["seed"], "policy": task["policy"],
                     "cycle": task["cycle"], "status": "complete",
                     "macro_f1": metrics["f1_macro"], "balanced_accuracy": metrics["balanced_accuracy"],
                     "log_loss": metrics["log_loss"], "trainable_parameters": training["trainable_parameters"],
                     "training_seconds": training["elapsed_seconds"]})
    out = args.work_root / "report.csv"
    fields = ("index", "mode", "seed", "policy", "cycle", "status", "macro_f1", "balanced_accuracy", "log_loss", "trainable_parameters", "training_seconds")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"5B: {sum(r['status']=='complete' for r in rows)}/{len(rows)} complete. {out}")


def prepare_c(args) -> None:
    policies = checked_names(args.policies, POLICIES, "policies")
    seeds = checked_names(args.seeds, SEEDS, "seeds")
    if args.mode not in MODES or args.initialization not in ("warm_start", "restart_from_baseline"):
        raise ValueError("5C requires one valid method and one valid initialization")
    if args.cycles < 1 or args.cycles > 9 or args.epochs < 1 or args.patience < 1:
        raise ValueError("Invalid cycle or epoch count")
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    resolved = DatasetRegistry.from_toml(args.registry).resolve_profile("astrominn_ztf")
    if not resolved.dataset.validation_manifest or not resolved.dataset.test_manifest:
        raise FileNotFoundError("Fixed historical evaluation manifests required")
    for seed in seeds:
        base = args.data_root / "runs" / BASE_SUITE / "shared" / f"seed_{seed}" / "bootstrap_training/best_vit_tower.pth"
        if not base.is_file():
            raise FileNotFoundError(base)
    exclusion = (args.persistent_root / "runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv")
    if not exclusion.is_file():
        raise FileNotFoundError(f"Task 4's mandatory external-label exclusion audit is missing: {exclusion}")
    expected_suite = f"astrominn_rep5c_{args.mode}_{args.initialization}_v1"
    if args.suite_id != expected_suite:
        raise ValueError(f"Use collision-proof suite ID {expected_suite}")
    plan = {
        "schema": "representation_5c_v1", "suite_id": args.suite_id,
        "mode": args.mode, "initialization": args.initialization,
        "seeds": list(seeds), "policies": list(policies), "cycles": args.cycles,
        "registry": str(args.registry.resolve()), "registry_sha256": sha256(args.registry),
        "base_suite": BASE_SUITE, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "persistent_root": str(args.persistent_root), "data_root": str(args.data_root),
        "excluded_manifest": str(exclusion), "excluded_manifest_sha256": sha256(exclusion),
        "protocol": "Independent closed loop. Same original L0/U0, acquisition budget, validation/test, frozen experts; Vit adaptation mode is the only model adaptation factor.",
    }
    if not (args.repo / "scripts/run_representation_5c.py").exists():
        raise FileNotFoundError("Generate rep5bc 5C runner first")
    freeze(args.repo, args.work_root)
    dump(args.work_root / "plan.json", plan)
    print(f"5C prepared: {len(seeds)*len(policies)*args.cycles} cycle fits in {args.suite_id}. No jobs queued.")


def command_c(plan: dict, seed: int, policy: str, cycle: int, work: Path) -> list[str]:
    runner = work / "source" / "scripts/run_representation_5c.py"
    command = [
        sys.executable, str(runner), "--registry", plan["registry"],
        "--suite-id", plan["suite_id"], "--bootstrap-source-suite-id", plan["base_suite"],
        "--require-existing-bootstrap", "--training-initialization", plan["initialization"],
        "--representation-mode", plan["mode"], "--strategies", policy,
        "--seed", str(seed), "--split-seed", "135", "--initial-fraction", "0.10",
        "--minimum-initial-per-class", "5", "--acquisition-fraction", "0.10",
        "--max-cycles", str(plan["cycles"]),
        "--max-epochs", str(plan["epochs"]), "--patience", str(plan["patience"]),
        "--batch-size", str(plan["batch_size"]), "--validation-batch-size", "2048",
        "--learning-rate", str(plan["learning_rate"]),
        "--train-workers", "0", "--validation-workers", "0", "--inference-workers", "0",
        "--evaluation-workers", "0", "--cycle", str(cycle), "--run",
        "--assert-excluded-manifest", plan["excluded_manifest"],
    ]
    # For the final ninth cycle, the original U0 remainder is 469, not 473.
    if plan["cycles"] == 9:
        command.append("--complete-pool")
    return command


def run_c(args) -> None:
    plan = load_plan(args.work_root, "5c")
    if args.seed not in plan["seeds"] or args.policy not in plan["policies"] or not 1 <= args.cycle <= plan["cycles"]:
        raise ValueError("5C worker is not in its frozen experiment plan")
    env = os.environ.copy()
    path = str(args.work_root / "source/src")
    env["PYTHONPATH"] = path + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = command_c(plan, args.seed, args.policy, args.cycle, args.work_root)
    print("5C command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def report_c(args) -> None:
    """Audit 5C training/inference provenance and summarize validation curves."""
    plan = load_plan(args.work_root, "5c")
    rows = []
    for seed in plan["seeds"]:
        baseline = (Path(plan["data_root"]) / "runs" / plan["base_suite"] / "shared"
                    / f"seed_{seed}" / "bootstrap_training/best_vit_tower.pth")
        for policy in plan["policies"]:
            previous = baseline
            selected = set()
            for cycle in range(1, plan["cycles"] + 1):
                file = (Path(plan["persistent_root"]) / "runs" / plan["suite_id"] / policy
                        / f"seed_{seed}" / f"cycle_{cycle:03d}" / "cycle_summary.json")
                base = {"seed": seed, "policy": policy, "cycle": cycle}
                if not file.exists():
                    rows.append({**base, "status": "missing"})
                    continue
                record = json.loads(file.read_text())
                checkpoint = Path(record["trained_checkpoint"])
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing checkpoint after cycle {cycle}: {checkpoint}")
                if record.get("inference_checkpoint") != str(previous):
                    raise ValueError(f"Wrong inference checkpoint for {file}")
                expected_initial = previous if plan["initialization"] == "warm_start" else baseline
                if record.get("training_initial_checkpoint") != str(expected_initial):
                    raise ValueError(f"Wrong 5C training initialization for {file}")
                summary = Path(record["training_summary"])
                training = json.loads(summary.read_text())
                if training.get("initial_checkpoint") != str(expected_initial):
                    raise ValueError(f"Training summary disagrees with requested initialization: {summary}")
                ids = record.get("selected_ids")
                if not isinstance(ids, list) or len(ids) != len(set(ids)) or selected.intersection(ids):
                    raise ValueError(f"Duplicate object acquisitions: {file}")
                expected_batch = 469 if (cycle == 9 and plan["cycles"] == 9) else 473
                if len(ids) != expected_batch or record.get("acquisition_size") != expected_batch:
                    raise ValueError(f"Incorrect acquisition budget: {file}")
                selected.update(ids)
                evalfile = Path(record["evaluation_summary"])
                evaluation = json.loads(evalfile.read_text())
                validation = next(item for item in evaluation["splits"] if item["split"] == "validation")
                metrics = validation["object_metrics"]
                rows.append({**base, "status": "complete", "checkpoint": str(checkpoint),
                             "macro_f1": metrics["f1_macro"],
                             "balanced_accuracy": metrics["balanced_accuracy"],
                             "log_loss": metrics["log_loss"],
                             "training_seconds": training["elapsed_seconds"]})
                previous = checkpoint
    out = args.work_root / "report.csv"
    fields = ("seed", "policy", "cycle", "status", "checkpoint", "macro_f1",
              "balanced_accuracy", "log_loss", "training_seconds")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"5C: {sum(r['status']=='complete' for r in rows)}/{len(rows)} audited cycles. {out}")


def submit_b(args) -> None:
    plan = load_plan(args.work_root, "5b")
    if args.parallel < 1:
        raise ValueError("--parallel must be positive")
    for task in plan["tasks"]:  # all before scheduler call
        verify_b_task(plan, task)
    worker = args.work_root / "source/scripts/adaptive/rep5bc_worker.sbatch"
    num = len(plan["tasks"])
    command = ["sbatch", "--hold", "--parsable", f"--array=0-{num-1}%{args.parallel}", f"--time={args.walltime}",
               f"--output={args.work_root}/logs/5b_%A_%a.out", str(worker), "5b", str(args.work_root)]
    print("5B submission:", " ".join(command))
    if not args.submit:
        print("DRY RUN — no jobs queued")
        return
    marker = args.work_root / "submitted.json"
    if marker.exists():
        raise FileExistsError(marker)
    (args.work_root / "logs").mkdir(exist_ok=True)
    job = subprocess.check_output(command, text=True).strip()
    if not job.isdecimal():
        raise ValueError(f"Unexpected SLURM job id: {job}")
    try:
        dump(marker, {"job_id": job, "count": num})
        subprocess.run(["scontrol", "release", job], check=True)
    except Exception:
        subprocess.run(["scancel", job], check=False)
        marker.unlink(missing_ok=True)
        raise
    print("5B submitted SLURM array", job)


def submit_c(args) -> None:
    plan = load_plan(args.work_root, "5c")
    marker = args.work_root / "submitted.json"
    if args.submit and marker.exists():
        raise FileExistsError(marker)
    worker = args.work_root / "source/scripts/adaptive/rep5bc_worker.sbatch"
    job_plan = []
    for seed in plan["seeds"]:
        for policy in plan["policies"]:
            for cycle in range(1, plan["cycles"] + 1):
                job_plan.append((seed, policy, cycle))
    print(f"5C: {len(job_plan)} jobs; sequential chain per seed; maximum 3 independent seed chains")
    print("DRY RUN" if not args.submit else "SUBMITTING")
    if not args.submit:
        for job in job_plan[:6]:
            print("  ", job)
        return
    logdir = args.work_root / "logs"
    logdir.mkdir(exist_ok=True)
    queued = []
    last_by_seed = {}
    try:
        for seed, policy, cycle in job_plan:
            deps = ([f"--dependency=afterok:{last_by_seed[seed]}"] if seed in last_by_seed else [])
            command = ["sbatch", "--hold", "--parsable", "--time", args.walltime,
                       f"--output={logdir}/c_{seed}_{policy}_{cycle}_%j.out", *deps,
                       str(worker), "5c", str(args.work_root), str(seed), policy, str(cycle)]
            jid = subprocess.check_output(command, text=True).strip()
            if not jid.isdecimal():
                raise ValueError(f"Not a numeric SLURM job id: {jid}")
            queued.append((seed, policy, cycle, jid))
            last_by_seed[seed] = jid
        dump(marker, {"jobs": [{"seed": s, "policy": p, "cycle": c, "job_id": j} for s,p,c,j in queued]})
        # Release from end to start so no downstream edge is missed.
        for _, _, _, jid in reversed(queued):
            subprocess.run(["scontrol", "release", jid], check=True)
    except Exception:
        for _, _, _, jid in queued:
            subprocess.run(["scancel", jid], check=False)
        marker.unlink(missing_ok=True)
        raise
    print(f"Submitted {len(queued)} 5C jobs, {len(last_by_seed)} independent chains")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare-b", "run-b", "report-b", "prepare-c", "run-c", "report-c", "submit-b", "submit-c"):
        p = sub.add_parser(name)
        p.add_argument("--work-root", type=Path, required=True)
        if name.startswith("prepare"):
            p.add_argument("--repo", type=Path, default=REPO)
            p.add_argument("--registry", type=Path, required=True)
            p.add_argument("--persistent-root", type=Path, default=PROJECT)
            p.add_argument("--data-root", type=Path, default=DATA)
            p.add_argument("--policies", nargs="+", default=list(POLICIES))
            p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
            p.add_argument("--epochs", type=int, default=50)
            p.add_argument("--patience", type=int, default=10)
            p.add_argument("--batch-size", type=int, default=512)
            p.add_argument("--learning-rate", type=float, default=1e-4)
        if name == "prepare-b":
            p.add_argument("--source-suite", required=True)
            p.add_argument("--cycles", nargs="+", type=int, default=[1, 3, 5, 7])
        if name == "prepare-c":
            p.add_argument("--mode", choices=MODES, required=True)
            p.add_argument("--initialization", choices=("warm_start", "restart_from_baseline"), required=True)
            p.add_argument("--suite-id", required=True)
            p.add_argument("--cycles", type=int, default=5)
        if name == "run-b":
            p.add_argument("--task-index", type=int, required=True)
            p.add_argument("--device", default="cuda")
        if name == "run-c":
            p.add_argument("--seed", type=int, required=True)
            p.add_argument("--policy", required=True)
            p.add_argument("--cycle", type=int, required=True)
        if name.startswith("submit"):
            p.add_argument("--submit", action="store_true")
            p.add_argument("--walltime", default="00:40:00")
            if name == "submit-b":
                p.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()
    if args.action == "prepare-b": prepare_b(args)
    elif args.action == "run-b": run_b(args)
    elif args.action == "report-b": report_b(args)
    elif args.action == "prepare-c": prepare_c(args)
    elif args.action == "run-c": run_c(args)
    elif args.action == "report-c": report_c(args)
    elif args.action == "submit-b": submit_b(args)
    elif args.action == "submit-c": submit_c(args)


if __name__ == "__main__":
    main()
