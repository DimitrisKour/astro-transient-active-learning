"""Controlled retraining of saved historical manifests, without acquisition.

Preparation uses the trainer label mapping to validate replay aliases.
Existing AL state is always read-only.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from al_pipeline.astronomy.data.registry import DatasetRegistry, DatasetSpec

ORDER_STRATEGIES = (
    "random", "entropy", "entropy_diversity", "entropy_balanced_explore",
)
ORDER_SEEDS = (135, 246, 357)
SCHEMA_VERSION = 1


def environment_versions() -> dict:
    result = {"python": platform.python_version()}
    for package in ("numpy", "torch", "torchvision", "scikit-learn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


@dataclass(frozen=True)
class Manifest:
    fields: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    object_ids: frozenset[str]
    content_sha256: str


def read_manifest(path: Path, dataset: DatasetSpec) -> Manifest:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required = {dataset.object_id_column, dataset.sample_file_column, dataset.label_column}
        if not fields or len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"Missing/duplicate manifest columns: {path}")
        rows = tuple(dict(row) for row in reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Empty or malformed manifest: {path}")
    labels: dict[str, str] = {}
    for row in rows:
        object_id = row[dataset.object_id_column]
        label = row[dataset.label_column]
        if not object_id or not label or not row[dataset.sample_file_column]:
            raise ValueError(f"Missing object, label or sample path: {path}")
        if labels.setdefault(object_id, label) != label:
            raise ValueError(f"Inconsistent native labels for {object_id}: {path}")
    columns = sorted(fields)
    content = sorted(tuple(row[c] for c in columns) for row in rows)
    return Manifest(fields, rows, frozenset(labels), json_hash([columns, content]))


def _model_content_hash(manifest: Manifest, dataset: DatasetSpec) -> str:
    """Allow only known label aliases and omitted replay survey metadata."""
    from al_pipeline.astronomy.astrominn.training import LABEL_TO_INDEX

    columns = sorted(manifest.fields)
    content = []
    for row in manifest.rows:
        normalized = dict(row)
        label = row[dataset.label_column]
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown trainer label: {label}")
        normalized[dataset.label_column] = str(LABEL_TO_INDEX[label])
        if "survey" in normalized:
            survey = normalized["survey"]
            if survey and survey != dataset.survey:
                raise ValueError(f"Unexpected survey: {survey}")
            normalized["survey"] = dataset.survey
        content.append(tuple(normalized[c] for c in columns))
    return json_hash([columns, sorted(content)])


def canonical_rows(manifest: Manifest, dataset: DatasetSpec) -> list[dict[str, str]]:
    """Preserve duplicate rows; acquisition history is not a sorting key."""
    def key(row: dict[str, str]) -> tuple:
        alert = row.get("alert", "0")
        return (
            row[dataset.object_id_column], int(alert),
            row[dataset.sample_file_column],
            tuple(row[c] for c in sorted(manifest.fields)),
        )
    return sorted(manifest.rows, key=key)


def _write_manifest(path: Path, manifest: Manifest, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest.fields)
        writer.writeheader()
        writer.writerows(rows)


def _sample_paths(manifest: Manifest, dataset: DatasetSpec) -> set[Path]:
    result = set()
    for row in manifest.rows:
        path = Path(row[dataset.sample_file_column])
        if not path.is_absolute() and dataset.data_root is not None:
            path = dataset.data_root / path
        result.add(path)
    return result


def _source_run(
    source_suite: Path, strategy: str, seed: int, cycle: int,
    dataset: DatasetSpec, immutable: dict[str, str], artifact_root: Path | None,
) -> tuple[Path, Manifest, dict]:
    state_dir = source_suite / strategy / f"seed_{seed}"
    experiment_path = state_dir / "experiment.json"
    state_path = state_dir / "run_state.json"
    experiment = json.loads(experiment_path.read_text())
    state = json.loads(state_path.read_text())
    if state.get("active_cycle") is not None or state.get("last_completed_cycle", 0) < cycle:
        raise ValueError(f"Source cycle is incomplete or active: {state_dir}, cycle {cycle}")
    if experiment["config"]["training_initialization"] != "restart_from_random":
        raise ValueError("Diagnostics require the completed random-ViT restart suite.")
    heavy = (
        artifact_root / strategy / f"seed_{seed}"
        if artifact_root is not None else Path(experiment["artifact_dir"])
    )
    summary_path = heavy / f"cycle_{cycle:03d}" / "training" / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    path = state_dir / "manifests" / f"train_cycle_{cycle:03d}.csv"
    manifest = read_manifest(path, dataset)
    expected = {
        "cycle": cycle, "seed": seed, "strategy": strategy,
        "initialization": "random_vit", "initial_checkpoint": None,
        "train_manifest_sha256": sha256(path), "training_samples": len(manifest.rows),
        "validation_manifest_sha256": sha256(dataset.validation_manifest),
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(f"Source training mismatch {name}: {summary_path}")
    policy = summary["policy"]
    if (
        policy.get("legacy_seed_sequence") or policy.get("legacy_frozen_expert_train_mode")
        or policy.get("class_weight_source") != "training_manifest"
        or policy.get("new_object_weighting") != "per_sample" or policy.get("overwrite")
    ):
        raise ValueError(f"Unsupported saved training recipe: {summary_path}")
    if experiment["validation_manifest_sha256"] != sha256(dataset.validation_manifest):
        raise ValueError("Registry validation data differ from the completed experiment.")
    for source in [experiment_path, state_path, summary_path, path]:
        immutable[str(source.resolve())] = sha256(source)
    return path, manifest, summary


def _policy_without_loader(policy: dict) -> dict:
    return {key: value for key, value in policy.items() if key != "loader_seed"}


def _freeze_source(repo: Path, destination: Path) -> dict[str, str]:
    files = [
        *sorted((repo / "src" / "al_pipeline").rglob("*.py")),
        repo / "scripts/run_astrominn_training_diagnostic.py",
        repo / "scripts/report_training_diagnostics.py",
        repo / "scripts/adaptive/training_diagnostic.sbatch",
    ]
    hashes = {}
    for source in files:
        target = destination / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(target.relative_to(destination))] = sha256(target)
    return hashes


def prepare_diagnostics(
    *, repo: Path, registry_path: Path, diagnostic_id: str,
    source_suite: Path | None = None, source_artifact_root: Path | None = None,
    persistent_root: Path | None = None, data_root: Path | None = None,
    trajectory_strategy: str = "entropy_balanced_explore", trajectory_seed: int = 135,
    repeat_count: int = 20, first_repeat_seed: int = 1001,
    repeat_loader_seed: int | None = None,
    legacy_code_dir: Path = Path("/projects/bcrv/ffontinelenunes/things_for_hailey"),
    profile_name: str = "astrominn_ztf",
) -> Path:
    """Freeze inputs, source code and 57 independent fits at the default settings."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", diagnostic_id):
        raise ValueError("Invalid diagnostic ID.")
    if trajectory_strategy not in ORDER_STRATEGIES or trajectory_seed < 0:
        raise ValueError("Invalid trajectory strategy/seed.")
    if repeat_count < 2 or first_repeat_seed < 0:
        raise ValueError("Use at least two nonnegative repeat seeds.")
    if repeat_loader_seed is not None and repeat_loader_seed < 0:
        raise ValueError("Loader seed must be nonnegative.")
    seeds = list(range(first_repeat_seed, first_repeat_seed + repeat_count))
    if trajectory_seed in seeds:
        raise ValueError("Fresh repeat seeds must exclude the original anchor seed.")
    registry_path = registry_path.resolve()
    resolved = DatasetRegistry.from_toml(registry_path).resolve_profile(profile_name)
    dataset = resolved.dataset
    persistent_root = (persistent_root or resolved.storage.persistent_root).resolve()
    data_root = (data_root or resolved.storage.data_root).resolve()
    source_suite = (
        source_suite or resolved.storage.persistent_root / "runs/astrominn_historical_replay_v1"
    ).resolve()
    if source_artifact_root is not None:
        source_artifact_root = source_artifact_root.resolve()
    legacy_code_dir = legacy_code_dir.resolve()
    metadata = persistent_root / "training_diagnostics" / diagnostic_id
    artifacts = data_root / "training_diagnostics" / diagnostic_id
    if metadata.is_relative_to(source_suite) or artifacts.is_relative_to(source_suite):
        raise ValueError("Diagnostic outputs must be separate from the source AL suite.")
    identity = {
        "registry": str(registry_path), "diagnostic_id": diagnostic_id,
        "source_suite": str(source_suite),
        "source_artifact_root": str(source_artifact_root) if source_artifact_root else None,
        "persistent_root": str(persistent_root), "data_root": str(data_root),
        "trajectory_strategy": trajectory_strategy, "trajectory_seed": trajectory_seed,
        "repeat_seeds": seeds, "repeat_loader_seed": repeat_loader_seed,
        "legacy_code_dir": str(legacy_code_dir), "profile": profile_name,
    }
    plan_path = metadata / "plan.json"
    if plan_path.exists():
        plan = load_plan(plan_path)
        if plan["identity"] != identity:
            raise ValueError("Prepared diagnostic settings differ; use a new diagnostic ID.")
        verify_plan(plan_path)
        return plan_path
    if metadata.exists():
        raise ValueError(f"Incomplete preparation already exists: {metadata}; use a new ID.")
    full = read_manifest(dataset.train_manifest, dataset)
    validation = read_manifest(dataset.validation_manifest, dataset)
    test = read_manifest(dataset.test_manifest, dataset)
    if (
        full.object_ids & validation.object_ids or full.object_ids & test.object_ids
        or validation.object_ids & test.object_ids
    ):
        raise ValueError("Historical train/validation/test objects overlap.")
    immutable = {
        str(path.resolve()): sha256(path)
        for path in [registry_path, dataset.train_manifest,
                     dataset.validation_manifest, dataset.test_manifest]
    }
    artifact_dir = Path(resolved.profile.options["checkpoint"]).parent
    for name in (
        "astroMiNN_config.json", "best_lc_tower_135.pth",
        "best_spatial_tower_135.pth", "best_coord_tower_135.pth",
    ):
        path = artifact_dir / name
        immutable[str(path.resolve())] = sha256(path)
    # The model imports modules directly from this directory. Wandb archives
    # below it may contain dangling source links and are not runtime inputs.
    legacy_sources = sorted(legacy_code_dir.glob("*.py"))
    if not (legacy_code_dir / "astroMiNN_V2.py").is_file():
        raise FileNotFoundError(legacy_code_dir / "astroMiNN_V2.py")
    for path in legacy_sources:
        immutable[str(path)] = sha256(path)
    samples = _sample_paths(full, dataset) | _sample_paths(validation, dataset)
    samples |= _sample_paths(test, dataset)
    missing = [str(path) for path in samples if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing historical sample files (first 10): {missing[:10]}")
    historical_content = _model_content_hash(full, dataset)
    replay_full = None
    order_sources = {}
    common_policy = None
    policy_by_seed = {}
    for seed in ORDER_SEEDS:
        for strategy in ORDER_STRATEGIES:
            path, manifest, summary = _source_run(
                source_suite, strategy, seed, 9, dataset, immutable, source_artifact_root,
            )
            if _model_content_hash(manifest, dataset) != historical_content:
                raise ValueError(f"Full-data row content differs from the historical dataset: {path}")
            if replay_full is None:
                replay_full = manifest
            elif manifest.content_sha256 != replay_full.content_sha256:
                raise ValueError(f"Full-data replay manifests have different raw row content: {path}")
            base = _policy_without_loader(summary["policy"])
            if common_policy is not None and common_policy != base:
                raise ValueError("Full-data source training recipes differ beyond loader seed.")
            common_policy = base
            previous = policy_by_seed.setdefault(seed, summary["policy"])
            if previous != summary["policy"]:
                raise ValueError("Order variants must also use the same loader seed per model seed.")
            order_sources[seed, strategy] = (path, manifest, summary)
    # All order controls use exactly the saved replay rows, including metadata.
    full = replay_full
    transition_sources = {}
    for cycle in (6, 7):
        source = _source_run(
            source_suite, trajectory_strategy, trajectory_seed, cycle,
            dataset, immutable, source_artifact_root,
        )
        if _policy_without_loader(source[2]["policy"]) != common_policy:
            raise ValueError("Cycle-6/7 recipe differs from the full-data recipe.")
        transition_sources[cycle] = source
    l6, l7 = transition_sources[6][1], transition_sources[7][1]
    if not l6.object_ids < l7.object_ids:
        raise ValueError("Cycle 7 must strictly extend cycle 6's object set.")
    c6 = Counter(tuple(row[c] for c in sorted(l6.fields)) for row in l6.rows)
    c7 = Counter(tuple(row[c] for c in sorted(l7.fields)) for row in l7.rows)
    if l6.fields != l7.fields or c6 - c7:
        raise ValueError("Cycle 7 removed or modified existing sample rows.")
    for cycle, (_, manifest, _) in transition_sources.items():
        if not manifest.object_ids.issubset(full.object_ids):
            raise ValueError(f"Cycle {cycle} includes non-historical objects.")
        if manifest.object_ids & (validation.object_ids | test.object_ids):
            raise ValueError("Trajectory contains evaluation objects.")
    if transition_sources[6][2]["policy"] != transition_sources[7][2]["policy"]:
        raise ValueError("Cycle-6/7 training recipes differ.")
    evaluation_path = source_suite / trajectory_strategy / f"seed_{trajectory_seed}" / "evaluation_history.json"
    anchor_evaluations = [r for r in json.loads(evaluation_path.read_text())
                          if r["cycle"] in (6, 7) and r["split"] in ("validation", "test")]
    if len(anchor_evaluations) != 4:
        raise ValueError("Original cycle-6/7 validation/test evaluations are missing or duplicated.")
    immutable[str(evaluation_path)] = sha256(evaluation_path)
    # All source checks precede preparation writes. Never rewrite source manifests.
    for path in [metadata / "logs", artifacts]:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b"diagnostic write probe")
            probe.flush()
            os.fsync(probe.fileno())
    shutil.copyfile(registry_path, metadata / "registry.toml")
    manifests = metadata / "manifests"
    canonical_path = manifests / "full_canonical.csv"
    _write_manifest(canonical_path, full, canonical_rows(full, dataset))
    if read_manifest(canonical_path, dataset).content_sha256 != full.content_sha256:
        raise ValueError("Canonical sorting changed sample content/multiplicities.")
    copied = {}
    for cycle, (path, manifest, summary) in transition_sources.items():
        target = manifests / f"trajectory_cycle_{cycle:03d}_original.csv"
        shutil.copyfile(path, target)
        copied["transition", cycle] = target
    for (seed, strategy), (path, manifest, summary) in order_sources.items():
        target = manifests / f"full_{strategy}_seed_{seed}_original.csv"
        shutil.copyfile(path, target)
        copied[seed, strategy] = target
    tasks = []
    def add(kind: str, cycle: int, seed: int, variant: str, manifest_path: Path,
            summary: dict, anchor: bool = False) -> None:
        policy = dict(summary["policy"])
        if kind == "transition" and not anchor and repeat_loader_seed is not None:
            policy["loader_seed"] = repeat_loader_seed
        manifest = read_manifest(manifest_path, dataset)
        index = len(tasks)
        tasks.append({
            "index": index, "task_id": f"fit_{index:03d}", "kind": kind,
            "cycle": cycle, "training_seed": seed, "variant": variant,
            "anchor": anchor, "loader_seed": policy["loader_seed"], "policy": policy,
            "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "content_sha256": manifest.content_sha256,
            "objects": len(manifest.object_ids), "samples": len(manifest.rows),
            "source_run_id": summary["run_id"],
        })
    for cycle in (6, 7):
        add("transition", cycle, trajectory_seed, "original_order", copied["transition", cycle],
            transition_sources[cycle][2], anchor=True)
    for seed in seeds:
        for cycle in (6, 7):
            add("transition", cycle, seed, "original_order", copied["transition", cycle],
                transition_sources[cycle][2])
    for seed in ORDER_SEEDS:
        add("order", 9, seed, "canonical", canonical_path, order_sources[seed, "random"][2])
        for strategy in ORDER_STRATEGIES:
            add("order", 9, seed, strategy, copied[seed, strategy], order_sources[seed, strategy][2])
    source_hashes = _freeze_source(repo.resolve(), metadata / "source")
    # Frozen code can run while the working checkout is edited for later experiments.
    plan = {
        "schema_version": SCHEMA_VERSION, "identity": identity,
        "metadata_root": str(metadata), "artifact_root": str(artifacts),
        "registry": str(metadata / "registry.toml"), "profile": profile_name,
        "legacy_code_dir": str(legacy_code_dir), "model_artifact_dir": str(artifact_dir),
        "immutable_inputs": immutable, "source_hashes": source_hashes,
        "registry_sha256": sha256(metadata / "registry.toml"),
        "tasks": tasks, "repeat_seeds": seeds,
        "anchor_source_evaluations": anchor_evaluations,
        "order_strategies": list(ORDER_STRATEGIES),
        "environment_versions": environment_versions(),
        "full_data": {"objects": len(full.object_ids), "samples": len(full.rows),
                      "content_sha256": full.content_sha256},
        "original_runtime_seconds": {
            str(cycle): transition_sources[cycle][2]["elapsed_seconds"] for cycle in (6, 7)
        },
        "interpretation": {
            "transition": "Fixed selected objects and original row order; vary training seed, "
                          "including initialization/dropout/augmentation RNG; fixed loader seed.",
            "order": "Same full sample content, training seed, loader seed and recipe; "
                     "only manifest row order changes.",
            "checkpoint": "Original weighted sample-level validation-loss rule; test is report-only.",
        },
    }
    write_json(plan_path, plan)
    (metadata / "plan.sha256").write_text(sha256(plan_path) + "\n")
    write_json(metadata / "preflight.json", {
        "passed": True, "task_count": len(tasks), "source_suite": str(source_suite),
        "source_files_hashed": len(immutable), "sample_files_checked": len(samples),
        "full_data": plan["full_data"], "new_objects_cycle_7": len(l7.object_ids - l6.object_ids),
        "training_only": True, "acquisition_executed": False,
    })
    verify_plan(plan_path)
    return plan_path


def load_plan(plan_path: Path) -> dict:
    plan_path = Path(plan_path).resolve()
    expected = (plan_path.parent / "plan.sha256").read_text().strip()
    if sha256(plan_path) != expected:
        raise ValueError("Prepared diagnostic plan changed.")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported diagnostic plan version.")
    return plan


def verify_plan(plan_path: Path, task_index: int | None = None) -> dict:
    plan = load_plan(plan_path)
    source_root = Path(plan["metadata_root"]) / "source"
    checks = dict(plan["immutable_inputs"])
    checks[plan["registry"]] = plan["registry_sha256"]
    checks.update({str(source_root / name): digest for name, digest in plan["source_hashes"].items()})
    tasks = plan["tasks"] if task_index is None else [plan["tasks"][task_index]]
    checks.update({task["manifest"]: task["manifest_sha256"] for task in tasks})
    for name, digest in checks.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Prepared diagnostic input/source changed: {name}")
    return plan


def task_result_path(plan: dict, task: dict) -> Path:
    return Path(plan["metadata_root"]) / "tasks" / task["task_id"] / "result.json"


def run_task(
    plan_path: Path, task_index: int, *, device: str = "cuda",
    trainer_class: Any = None, evaluator_class: Any = None,
) -> Path:
    """One independently schedulable fit. Resume completed training/evaluation safely."""
    import fcntl  # Training runs on Linux; metadata reporting also works on Windows.
    if task_index < 0:
        raise ValueError("Task index must be nonnegative.")
    plan = verify_plan(plan_path, task_index)
    if environment_versions() != plan["environment_versions"]:
        raise ValueError("Python/package versions differ from the prepared diagnostic environment.")
    task = plan["tasks"][task_index]
    metadata = task_result_path(plan, task).parent
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "fit.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Diagnostic task already running: {task['task_id']}") from exc
        return _run_locked(plan_path, plan, task, metadata, device, trainer_class, evaluator_class)


def _run_locked(plan_path: Path, plan: dict, task: dict, metadata: Path,
                device: str, trainer_class: Any, evaluator_class: Any) -> Path:
    from al_pipeline.astronomy.astrominn.training import (
        AstroMiNNViTTrainer, AstroMiNNViTTrainingConfig,
    )
    from al_pipeline.astronomy.astrominn.evaluation import AstroMiNNViTEvaluator
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import load_astrominn_vit_model
    from al_pipeline.core.training.base import TrainingRequest
    import torch

    trainer_class = trainer_class or AstroMiNNViTTrainer
    evaluator_class = evaluator_class or AstroMiNNViTEvaluator
    policy_data = dict(task["policy"])
    policy_data["class_count_reference"] = tuple(policy_data["class_count_reference"])
    policy = AstroMiNNViTTrainingConfig(**policy_data)
    dataset = DatasetRegistry.from_toml(plan["registry"]).resolve_profile(plan["profile"]).dataset
    initialization_path = metadata / "initialization.json"
    def audited_model_factory(load_checkpoint: bool, checkpoint: Path | None):
        model = load_astrominn_vit_model(
            legacy_code_dir=Path(plan["legacy_code_dir"]),
            artifact_dir=Path(plan["model_artifact_dir"]),
            load_vit_checkpoint=load_checkpoint, vit_checkpoint=checkpoint,
        )
        digest = hashlib.sha256()
        for name, tensor in sorted(model.vit_tower.state_dict().items()):
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1)
                          .view(torch.uint8).numpy().tobytes())
        write_json(initialization_path, {
            "vit_state_sha256": digest.hexdigest(), "training_seed": task["training_seed"],
            "loader_seed": task["loader_seed"], "loaded_vit_checkpoint": load_checkpoint,
        })
        return model
    trainer = trainer_class(
        dataset=dataset, policy=policy, device=device, initialization="random_vit",
        legacy_code_dir=Path(plan["legacy_code_dir"]),
        artifact_dir=Path(plan["model_artifact_dir"]),
        model_factory=audited_model_factory,
    )
    heavy = Path(plan["artifact_root"]) / task["task_id"]
    result_path = metadata / "result.json"
    plan_digest = sha256(plan_path)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("plan_sha256") != plan_digest or result.get("task") != task:
            raise ValueError("Completed diagnostic belongs to a different plan/task.")
        checkpoint = Path(result["checkpoint_path"])
        if sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("Completed diagnostic checkpoint changed.")
        return result_path
    attempts = sorted(heavy.glob("attempt_*")) if heavy.exists() else []
    attempt = attempts[-1] if attempts else heavy / "attempt_001"
    def request_at(path: Path) -> TrainingRequest:
        # TODO: add an initialization-only RNG control if a later ablation must
        # hold dropout/augmentation randomness fixed while changing model weights.
        return TrainingRequest(
            train_manifest=Path(task["manifest"]), validation_manifest=dataset.validation_manifest,
            output_dir=path / "training",
            run_id=f"{plan['identity']['diagnostic_id']}_{task['task_id']}",
            cycle=task["cycle"], strategy="training_diagnostic", seed=task["training_seed"],
            initial_checkpoint=None,
        )
    request = request_at(attempt)
    completed = trainer.completed_result(request) if attempts else None
    if attempts and completed is None:
        # Epoch resume is not provided by the existing trainer. Retain partial
        # attempts, rather than overwriting them or pretending to resume epochs.
        attempt = heavy / f"attempt_{len(attempts) + 1:03d}"
        request = request_at(attempt)
    write_json(metadata / "status.json", {
        "status": "running", "task": task["task_id"], "attempt": str(attempt),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_unix": time.time(),
    })
    start = time.perf_counter()
    try:
        training = completed or trainer.train(request)
        evaluator = evaluator_class(
            dataset=dataset, device=device, batch_size=policy.validation_batch_size,
            num_workers=policy.validation_num_workers,
            legacy_code_dir=Path(plan["legacy_code_dir"]),
            artifact_dir=Path(plan["model_artifact_dir"]),
        )
        evaluation = evaluator.evaluate(
            checkpoint_path=training.checkpoint_path, output_dir=attempt / "evaluation",
        )
        per_class = {}
        for split in evaluation.splits:
            path = attempt / "evaluation" / f"{split.split}_per_class_metrics.json"
            if path.exists():
                per_class[split.split] = json.loads(path.read_text())["object"]
        shutil.copyfile(training.history_path, metadata / "training_history.json")
        training_summary = json.loads(training.summary_path.read_text())
        result = {
            "plan_sha256": plan_digest, "task": task,
            "checkpoint_path": str(training.checkpoint_path),
            "checkpoint_sha256": sha256(training.checkpoint_path),
            "training": training_summary, "evaluation": [asdict(s) for s in evaluation.splits],
            "object_per_class": per_class, "wall_seconds_this_attempt": time.perf_counter() - start,
            "initialization": json.loads(initialization_path.read_text())
                              if initialization_path.exists() else None,
            "environment": {
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "device": str(device),
                "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        }
        write_json(result_path, result)
        write_json(metadata / "status.json", {"status": "completed", "attempt": str(attempt)})
        return result_path
    except BaseException as exc:
        write_json(metadata / "status.json", {
            "status": "failed", "attempt": str(attempt), "error": repr(exc),
        })
        raise

