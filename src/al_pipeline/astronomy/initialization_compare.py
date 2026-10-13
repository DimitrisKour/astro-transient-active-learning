"""Fixed-trajectory B initialization ablation; never run acquisition here."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from al_pipeline.astronomy.data.registry import DatasetRegistry
from al_pipeline.astronomy.training_diagnostics import (
    ORDER_SEEDS, ORDER_STRATEGIES, _model_content_hash, _policy_without_loader,
    _sample_paths, _source_run, environment_versions, load_plan, read_manifest,
    sha256, task_result_path, verify_plan, write_json,
)

MODES = ("restart_from_l0", "warm_start")


def _row_counts(manifest):
    return Counter(tuple(row[c] for c in sorted(manifest.fields)) for row in manifest.rows)


def _validated_evaluation(payload: dict, dataset, validation, test) -> list[dict]:
    splits = payload["splits"]
    if len(splits) != 2 or {s["split"] for s in splits} != {"validation", "test"}:
        raise ValueError("Source evaluation must contain exactly validation and test.")
    for split in splits:
        manifest = validation if split["split"] == "validation" else test
        if (split["sample_count"], split["object_count"]) != (len(manifest.rows), len(manifest.object_ids)):
            raise ValueError("Source evaluation counts differ from the fixed registry split.")
        for name in ("f1_macro", "balanced_accuracy", "accuracy"):
            value = float(split["object_metrics"][name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid source object metric.")
    return splits


def prepare_comparison(
    *, repo: Path, registry_path: Path, comparison_id: str = "astrominn_initialization_compare_v1",
    source_suite: Path | None = None, source_artifact_root: Path | None = None,
    persistent_root: Path | None = None, data_root: Path | None = None,
    seeds: tuple[int, ...] = ORDER_SEEDS, max_cycle: int = 9,
    legacy_code_dir: Path = Path("/projects/bcrv/ffontinelenunes/things_for_hailey"),
    profile_name: str = "astrominn_ztf",
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", comparison_id):
        raise ValueError("Invalid comparison ID.")
    seeds = tuple(seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in ORDER_SEEDS for s in seeds):
        raise ValueError("Use distinct source seeds from 135, 246, 357.")
    if not 1 <= max_cycle <= 9:
        raise ValueError("max_cycle must be between 1 and 9.")
    repo, registry_path = repo.resolve(), registry_path.resolve()
    resolved = DatasetRegistry.from_toml(registry_path).resolve_profile(profile_name)
    dataset = resolved.dataset
    persistent_root = (persistent_root or resolved.storage.persistent_root).resolve()
    data_root = (data_root or resolved.storage.data_root).resolve()
    source_suite = (source_suite or resolved.storage.persistent_root / "runs/astrominn_historical_replay_v1").resolve()
    source_artifact_root = source_artifact_root.resolve() if source_artifact_root else None
    legacy_code_dir = legacy_code_dir.resolve()
    metadata = persistent_root / "initialization_compare" / comparison_id
    artifacts = data_root / "initialization_compare" / comparison_id
    for output in (metadata, artifacts):
        if output.is_relative_to(source_suite) or (source_artifact_root and output.is_relative_to(source_artifact_root)):
            raise ValueError("Comparison outputs overlap the source experiment.")
    identity = {
        "comparison_id": comparison_id, "registry": str(registry_path), "seeds": list(seeds),
        "max_cycle": max_cycle, "source_suite": str(source_suite),
        "source_artifact_root": str(source_artifact_root) if source_artifact_root else None,
        "persistent_root": str(persistent_root), "data_root": str(data_root),
        "legacy_code_dir": str(legacy_code_dir), "profile": profile_name,
    }
    plan_path = metadata / "plan.json"
    if plan_path.exists():
        plan = verify_plan(plan_path)
        if plan.get("kind") != "initialization_compare" or plan["identity"] != identity:
            raise ValueError("Prepared comparison settings differ; use another comparison ID.")
        return plan_path
    if metadata.exists():
        raise ValueError(f"Incomplete preparation exists; use another comparison ID: {metadata}")

    full = read_manifest(dataset.train_manifest, dataset)
    validation = read_manifest(dataset.validation_manifest, dataset)
    test = read_manifest(dataset.test_manifest, dataset)
    if full.object_ids & (validation.object_ids | test.object_ids) or validation.object_ids & test.object_ids:
        raise ValueError("Historical train/validation/test objects overlap.")
    immutable = {str(p.resolve()): sha256(p) for p in
                 (registry_path, dataset.train_manifest, dataset.validation_manifest, dataset.test_manifest)}
    artifact_dir = Path(resolved.profile.options["checkpoint"]).parent
    for name in ("astroMiNN_config.json", "best_lc_tower_135.pth", "best_spatial_tower_135.pth", "best_coord_tower_135.pth"):
        p = artifact_dir / name
        immutable[str(p.resolve())] = sha256(p)
    if not (legacy_code_dir / "astroMiNN_V2.py").is_file():
        raise FileNotFoundError(legacy_code_dir / "astroMiNN_V2.py")
    for p in sorted(legacy_code_dir.glob("*.py")):
        immutable[str(p)] = sha256(p)
    samples = _sample_paths(full, dataset) | _sample_paths(validation, dataset) | _sample_paths(test, dataset)
    missing = [str(p) for p in samples if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sample files (first 10): {missing[:10]}")

    sources, bootstraps, references = {}, {}, []
    common_recipe = None
    full_replay_content = None
    common_l0 = None
    for seed in seeds:
        for strategy in ORDER_STRATEGIES:
            state = source_suite / strategy / f"seed_{seed}"
            experiment = json.loads((state / "experiment.json").read_text())
            heavy = source_artifact_root / strategy / f"seed_{seed}" if source_artifact_root else Path(experiment["artifact_dir"])
            initial_manifest = Path(experiment["historical_train_manifest"])
            if sha256(initial_manifest) != experiment["historical_train_manifest_sha256"]:
                raise ValueError("Source L0 manifest hash mismatch.")
            l0 = read_manifest(initial_manifest, dataset)
            if not l0.object_ids < full.object_ids or l0.object_ids & (validation.object_ids | test.object_ids):
                raise ValueError("L0 must be a proper subset of historical training objects.")
            if common_l0 is not None and common_l0 != sha256(initial_manifest):
                raise ValueError("Source policies/seeds have different L0 row orders/content.")
            common_l0 = sha256(initial_manifest)
            immutable[str(initial_manifest.resolve())] = common_l0
            bootstrap_checkpoint = (source_artifact_root / "shared" / f"seed_{seed}" / "bootstrap_training/best_vit_tower.pth"
                                    if source_artifact_root else Path(experiment["baseline_checkpoint"]))
            bootstrap_summary = bootstrap_checkpoint.parent / "training_summary.json"
            bs = json.loads(bootstrap_summary.read_text())
            bootstrap_hash = sha256(bootstrap_checkpoint)
            expected = {"cycle": 0, "strategy": "bootstrap", "seed": seed, "initialization": "random_vit",
                        "initial_checkpoint": None, "train_manifest_sha256": common_l0,
                        "validation_manifest_sha256": sha256(dataset.validation_manifest), "training_samples": len(l0.rows)}
            if any(bs.get(k) != v for k, v in expected.items()) or bootstrap_hash != experiment["baseline_checkpoint_sha256"]:
                raise ValueError("Source bootstrap identity/checkpoint mismatch.")
            immutable[str(bootstrap_summary.resolve())] = sha256(bootstrap_summary)
            immutable[str(bootstrap_checkpoint.resolve())] = bootstrap_hash
            bootstrap = {"checkpoint": str(bootstrap_checkpoint), "sha256": bootstrap_hash,
                         "training": bs, "objects": len(l0.object_ids), "samples": len(l0.rows)}
            if seed in bootstraps and bootstraps[seed] != bootstrap:
                raise ValueError("Policies do not share the same seed's L0 checkpoint.")
            bootstraps[seed] = bootstrap
            history_path = state / "evaluation_history.json"
            history = json.loads(history_path.read_text())
            immutable[str(history_path.resolve())] = sha256(history_path)
            previous = l0
            for cycle in range(1, max_cycle + 1):
                path, manifest, training = _source_run(source_suite, strategy, seed, cycle, dataset, immutable, source_artifact_root)
                if not previous.object_ids < manifest.object_ids or not manifest.object_ids <= full.object_ids:
                    raise ValueError("Saved trajectory must strictly grow within historical training objects.")
                if set(previous.fields) != set(manifest.fields) or _row_counts(previous) - _row_counts(manifest):
                    raise ValueError("Saved trajectory removed or changed existing sample rows.")
                if manifest.object_ids & (validation.object_ids | test.object_ids):
                    raise ValueError("Saved trajectory includes evaluation objects.")
                recipe = _policy_without_loader(training["policy"])
                if common_recipe is not None and common_recipe != recipe:
                    raise ValueError("Source training recipes differ beyond loader seed.")
                common_recipe = recipe
                if training["policy"] != bs["policy"]:
                    raise ValueError("Bootstrap and source-cycle recipes/loader seeds differ.")
                if cycle == 9:
                    if _model_content_hash(manifest, dataset) != _model_content_hash(full, dataset):
                        raise ValueError("Full replay differs from historical model-equivalent rows.")
                    if full_replay_content is not None and full_replay_content != manifest.content_sha256:
                        raise ValueError("Final replay row content/multiplicities differ across policies.")
                    full_replay_content = manifest.content_sha256
                evaluation_path = heavy / f"cycle_{cycle:03d}/evaluation/evaluation_summary.json"
                evaluation = _validated_evaluation(json.loads(evaluation_path.read_text()), dataset, validation, test)
                immutable[str(evaluation_path.resolve())] = sha256(evaluation_path)
                for split in evaluation:
                    entries = [r for r in history if r["cycle"] == cycle and r["split"] == split["split"]]
                    if len(entries) != 1 or any(entries[0]["metrics"][m] != split["object_metrics"][m]
                                                for m in ("f1_macro", "accuracy", "balanced_accuracy")):
                        raise ValueError("Source evaluation history disagrees with saved evaluation.")
                sources[seed, strategy, cycle] = (path, manifest, training)
                references.append({"seed": seed, "policy": strategy, "cycle": cycle,
                                   "mode": "random_restart", "objects": len(manifest.object_ids),
                                   "samples": len(manifest.rows), "manifest_sha256": sha256(path),
                                   "training": training, "evaluation": evaluation})
                previous = manifest

    # No writes precede scientific/source checks. Copy saved rows without sorting.
    for directory in (metadata / "logs", artifacts):
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory) as probe:
            probe.write(b"initialization comparison write probe"); probe.flush(); os.fsync(probe.fileno())
    shutil.copyfile(registry_path, metadata / "registry.toml")
    manifest_paths = {}
    for (seed, strategy, cycle), (path, _, _) in sources.items():
        target = metadata / "manifests" / f"s{seed}_{strategy}_c{cycle:03d}.csv"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        manifest_paths[seed, strategy, cycle] = target
    tasks, lookup = [], {}
    for seed in seeds:
        for cycle in range(1, max_cycle + 1):
            for strategy in ORDER_STRATEGIES:
                path, manifest, summary = sources[seed, strategy, cycle]
                for mode in (("shared_l0_start",) if cycle == 1 else MODES):
                    task_id = f"s{seed}_{strategy}_{mode}_c{cycle:03d}"
                    predecessor = lookup[seed, strategy, cycle-1, "warm_start"] if cycle > 1 and mode == "warm_start" else None
                    copied = manifest_paths[seed, strategy, cycle]
                    task = {"index": len(tasks), "task_id": task_id, "seed": seed, "training_seed": seed,
                            "loader_seed": summary["policy"]["loader_seed"], "policy_name": strategy,
                            "cycle": cycle, "mode": mode, "modes": list(MODES) if cycle == 1 else [mode],
                            "policy": summary["policy"], "manifest": str(copied),
                            "manifest_sha256": sha256(copied), "content_sha256": manifest.content_sha256,
                            "objects": len(manifest.object_ids), "samples": len(manifest.rows),
                            "initial_task": predecessor,
                            "initial_checkpoint": bootstraps[seed]["checkpoint"] if predecessor is None else None,
                            "initial_checkpoint_sha256": bootstraps[seed]["sha256"] if predecessor is None else None}
                    tasks.append(task)
                    for effective in task["modes"]:
                        lookup[seed, strategy, cycle, effective] = task_id
    frozen = metadata / "source"
    files = [*sorted((repo / "src/al_pipeline").rglob("*.py")),
             repo / "scripts/run_astrominn_initialization_compare.py",
             repo / "scripts/report_initialization_compare.py",
             repo / "scripts/adaptive/initialization_cycle.sbatch"]
    hashes = {}
    for p in files:
        target = frozen / p.relative_to(repo); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target); hashes[str(target.relative_to(frozen))] = sha256(target)
    plan = {"schema_version": 1, "kind": "initialization_compare", "identity": identity,
            "metadata_root": str(metadata), "artifact_root": str(artifacts),
            "registry": str(metadata / "registry.toml"), "registry_sha256": sha256(metadata / "registry.toml"),
            "legacy_code_dir": str(legacy_code_dir), "model_artifact_dir": str(artifact_dir), "profile": profile_name,
            "immutable_inputs": immutable, "source_hashes": hashes, "environment_versions": environment_versions(),
            "tasks": tasks, "seeds": list(seeds), "strategies": list(ORDER_STRATEGIES), "max_cycle": max_cycle,
            "bootstraps": {str(s): b for s, b in bootstraps.items()}, "references": references,
            "full_data_objects": len(full.object_ids),
            "interpretation": {
                "acquisition": "Fixed saved batches from the original random-restart trajectories; no new acquisition.",
                "weights": "Explicit L0 checkpoint for restart; previous validation-selected ViT checkpoint for warm-start.",
                "optimizer": "Fresh optimizer and scheduler every fit, including warm-start; weights-only continuation.",
                "data": "Entire cumulative manifest, original row order and multiplicities; expert towers frozen.",
                "checkpoint": "Original weighted sample-level validation-loss selection; test is report-only.",
                "compute": "Same per-cycle epoch cap/patience; warm-start inherits earlier training, not compute-matched.",
                "scope": "Training comparison conditional on original policy trajectories; not new closed-loop policy performance.",
            }}
    write_json(plan_path, plan); (metadata / "plan.sha256").write_text(sha256(plan_path) + "\n")
    write_json(metadata / "preflight.json", {"passed": True, "new_fits": len(tasks),
               "existing_random_fits_reused": len(references), "existing_l0_checkpoints_reused": len(seeds),
               "shared_cycle_1_fits": 4 * len(seeds), "acquisition_executed": False})
    verify_plan(plan_path)
    return plan_path


def completed_result(plan_path: Path, task: dict, *, check_checkpoint: bool = True) -> dict | None:
    plan = load_plan(plan_path)
    path = task_result_path(plan, task)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("plan_sha256") != sha256(plan_path) or result.get("task") != task:
        raise ValueError(f"Completed fit belongs to a different plan/task: {path}")
    if check_checkpoint and sha256(Path(result["checkpoint_path"])) != result["checkpoint_sha256"]:
        raise ValueError(f"Completed fit checkpoint changed: {path}")
    return result


def _initial_checkpoint(plan_path: Path, plan: dict, task: dict) -> tuple[Path, str]:
    if task["initial_task"]:
        predecessor = next(t for t in plan["tasks"] if t["task_id"] == task["initial_task"])
        result = completed_result(plan_path, predecessor)
        if result is None:
            raise ValueError(f"Warm-start predecessor is not completed: {task['initial_task']}")
        return Path(result["checkpoint_path"]), result["checkpoint_sha256"]
    checkpoint = Path(task["initial_checkpoint"])
    if sha256(checkpoint) != task["initial_checkpoint_sha256"]:
        raise ValueError("L0 checkpoint changed.")
    return checkpoint, task["initial_checkpoint_sha256"]


def run_task(plan_path: Path, task_index: int, *, device: str = "cuda",
             trainer_class: Any = None, evaluator_class: Any = None, model_loader: Any = None) -> Path:
    import fcntl
    if task_index < 0:
        raise ValueError("Task index must be nonnegative.")
    plan = verify_plan(plan_path, task_index)
    if plan.get("kind") != "initialization_compare":
        raise ValueError("Wrong plan kind.")
    if environment_versions() != plan["environment_versions"]:
        raise ValueError("Python/package versions changed after preparation.")
    task = plan["tasks"][task_index]
    metadata = task_result_path(plan, task).parent; metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "fit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This fit is already running.") from exc
        if completed_result(plan_path, task):
            return metadata / "result.json"
        checkpoint, checkpoint_hash = _initial_checkpoint(plan_path, plan, task)
        return _run_locked(plan_path, plan, task, metadata, checkpoint, checkpoint_hash,
                           device, trainer_class, evaluator_class, model_loader)


def _run_locked(plan_path, plan, task, metadata, checkpoint, checkpoint_hash,
                device, trainer_class, evaluator_class, model_loader):
    import torch
    from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer, AstroMiNNViTTrainingConfig
    from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import load_astrominn_vit_model
    from al_pipeline.core.training.base import TrainingRequest

    trainer_class = trainer_class or AstroMiNNViTTrainer
    evaluator_class = evaluator_class or AstroMiNNViTEvaluator
    model_loader = model_loader or load_astrominn_vit_model
    policy_data = dict(task["policy"]); policy_data["class_count_reference"] = tuple(policy_data["class_count_reference"])
    policy = AstroMiNNViTTrainingConfig(**policy_data)
    dataset = DatasetRegistry.from_toml(plan["registry"]).resolve_profile(plan["profile"]).dataset
    def audited_factory(load_checkpoint, requested_checkpoint):
        if not load_checkpoint or Path(requested_checkpoint) != checkpoint:
            raise ValueError("Expected the explicit ablation checkpoint, not a published fallback.")
        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError("Initial checkpoint changed during training.")
        model = model_loader(legacy_code_dir=Path(plan["legacy_code_dir"]), artifact_dir=Path(plan["model_artifact_dir"]),
                             load_vit_checkpoint=True, vit_checkpoint=checkpoint)
        digest = hashlib.sha256()
        for name, tensor in sorted(model.vit_tower.state_dict().items()):
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        write_json(metadata / "initialization.json", {"loaded_vit_checkpoint": True,
                   "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                   "vit_state_sha256": digest.hexdigest(), "mode": task["mode"],
                   "predecessor_task": task["initial_task"], "optimizer_reset": True, "scheduler_reset": True})
        return model
    trainer = trainer_class(dataset=dataset, policy=policy, device=device, initialization="pretrained",
                            legacy_code_dir=Path(plan["legacy_code_dir"]), artifact_dir=Path(plan["model_artifact_dir"]),
                            model_factory=audited_factory)
    heavy = Path(plan["artifact_root"]) / task["task_id"]
    attempts = sorted(heavy.glob("attempt_*")) if heavy.exists() else []
    attempt = attempts[-1] if attempts else heavy / "attempt_001"
    def request_at(p):
        return TrainingRequest(train_manifest=Path(task["manifest"]), validation_manifest=dataset.validation_manifest,
             output_dir=p / "training", run_id=f"{plan['identity']['comparison_id']}_{task['task_id']}",
             cycle=task["cycle"], strategy=f"initialization_{task['mode']}", seed=task["seed"], initial_checkpoint=checkpoint)
    request = request_at(attempt)
    training = trainer.completed_result(request) if attempts else None
    if attempts and training is None:
        attempt = heavy / f"attempt_{len(attempts)+1:03d}"; request = request_at(attempt)
    start = time.perf_counter()
    write_json(metadata / "status.json", {"status": "running", "attempt": str(attempt),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_unix": time.time()})
    try:
        training = training or trainer.train(request)
        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError("Initial checkpoint changed during training.")
        initialization_path = metadata / "initialization.json"
        if not initialization_path.exists():
            raise ValueError("Missing audit of loaded initial weights.")
        initialization = json.loads(initialization_path.read_text())
        if initialization["checkpoint_sha256"] != checkpoint_hash or not initialization["loaded_vit_checkpoint"]:
            raise ValueError("Initialization audit does not match this fit.")
        evaluator = evaluator_class(dataset=dataset, device=device, batch_size=policy.validation_batch_size,
                                    num_workers=policy.validation_num_workers, legacy_code_dir=Path(plan["legacy_code_dir"]),
                                    artifact_dir=Path(plan["model_artifact_dir"]))
        evaluation = evaluator.evaluate(checkpoint_path=training.checkpoint_path, output_dir=attempt / "evaluation")
        splits = [asdict(s) for s in evaluation.splits]
        _validated_evaluation({"splits": splits}, dataset, read_manifest(dataset.validation_manifest, dataset), read_manifest(dataset.test_manifest, dataset))
        per_class = {}
        for s in evaluation.splits:
            p = attempt / "evaluation" / f"{s.split}_per_class_metrics.json"
            if p.exists(): per_class[s.split] = json.loads(p.read_text())["object"]
        shutil.copyfile(training.history_path, metadata / "training_history.json")
        write_json(metadata / "result.json", {"plan_sha256": sha256(plan_path), "task": task,
                   "checkpoint_path": str(training.checkpoint_path), "checkpoint_sha256": sha256(training.checkpoint_path),
                   "initialization": initialization, "training": json.loads(training.summary_path.read_text()),
                   "evaluation": splits, "object_per_class": per_class,
                   "wall_seconds_this_attempt": time.perf_counter()-start,
                   "environment": {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
                                   "slurm_job_id": os.environ.get("SLURM_JOB_ID")}})
        write_json(metadata / "status.json", {"status": "completed", "attempt": str(attempt)})
        return metadata / "result.json"
    except BaseException as exc:
        write_json(metadata / "status.json", {"status": "failed", "attempt": str(attempt), "error": repr(exc)})
        raise


def batch_selection(plan_path: Path, seed: int, cycle: int) -> list[int]:
    """Missing fits only; validate completed checkpoints before skipping them."""
    plan = load_plan(plan_path)
    return [t["index"] for t in plan["tasks"] if t["seed"] == seed and t["cycle"] == cycle
            and completed_result(plan_path, t) is None]
