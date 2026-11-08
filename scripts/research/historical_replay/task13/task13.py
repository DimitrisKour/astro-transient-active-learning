#!/usr/bin/env python3
"""Task13: frozen, 3-seed x 4-cycle TAILOR versus uniform meta-acquisition.

Preparation and verification do not start training, edit previous suites, or
submit Slurm jobs. GPU pilots execute acquisition only. Submission uses
--hold, and a separate explicit `release` is needed to start the experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile


SUITE = "astrominn_task13_tailor_pilot_v1"
POLICIES = ("tailor_diversity", "uniform_mixture")
ARMS = ("entropy_diversity", "entropy_balanced_explore", "fisher_batch")
SEEDS = (135, 246, 357)
CYCLES = 4
PERSISTENT = Path("/projects/bcrv/dkourtesis/applecider_adaptive")
DATA = Path("/work/hdd/bcrv/dkourtesis/applecider_adaptive")
JOB = PERSISTENT / "closed_loop" / SUITE
TASK4 = "astrominn_closed_loop_warm_v1"
TASK11 = "astrominn_task11_advanced_acquisition_v1"
TASK11_JOB = PERSISTENT / "closed_loop" / TASK11
BASE = "astrominn_historical_replay_v1"
EXCLUDE = PERSISTENT / "runs/boom_tns_new_labelled_inference_v1/hidden_labels.csv"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def relevant_source_hashes(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    return {
        str(file.relative_to(root)): sha256(file)
        for file in sorted(root.rglob("*"))
        if file.is_file() and file.suffix in {".py", ".sh", ".sbatch", ".toml"}
    }


def scientific_plan() -> dict:
    return {
        "suite": SUITE,
        "comparison": "TAILOR_Dirichlet_Thompson_vs_uniform_policy_mixture",
        "tailor_source": "Zhang_et_al_NeurIPS_2023_Algorithm_2",
        "reward": "1/(5*max(1,confirmed_object_count_class))",
        "posterior": "multiclass_Dirichlet_uniform_prior",
        "discount_factor": 0.9,
        "prior_concentration": 1.0,
        "posterior_feedback": "revealed_acquisition_labels_after_each_complete_batch_only",
        "arms": list(ARMS),
        "policies": list(POLICIES),
        "seeds": list(SEEDS),
        "acquisition_cycles": CYCLES,
        "training_jobs": len(POLICIES) * len(SEEDS) * CYCLES,
        "bootstrap_source_suite": BASE,
        "reference_suite": TASK4,
        "source_parent_suite": TASK11,
        "initial_fraction": 0.10,
        "minimum_initial_per_class": 5,
        "split_seed": 135,
        "acquisition_fraction": 0.10,
        "training_initialization": "warm_start",
        "max_epochs": 50,
        "patience": 10,
        "batch_size": 512,
        "validation_batch_size": 2048,
        "learning_rate": 0.0001,
        "workers": 0,
        "sample_weighting": "per_sample_historical_compatibility",
        "class_weights": "training_manifest_only",
        "evaluation": "frozen_validation_only",
        "frozen_test": "never_evaluate_during_policy_development",
        "external_exclusion": str(EXCLUDE),
        "release_policy": "held_until_explicit_release",
    }


def verify_source(directory: str | Path, manifest_path: str | Path) -> None:
    manifest = read_json(manifest_path)
    actual = relevant_source_hashes(directory)
    if manifest != actual:
        changed = sorted(
            file for file in set(actual) | set(manifest) if manifest.get(file) != actual.get(file)
        )
        raise ValueError(f"Frozen source no longer matches its manifest: {changed[:12]}")


def prepare(*, repo: str | Path, job_dir: str | Path = JOB) -> None:
    """Freeze Task11-vetted code plus the independently versioned Task13 overlay."""
    from task13_patch import patch_source

    job = Path(job_dir)
    repo = Path(repo).resolve()
    source_parent = TASK11_JOB
    verify_source(source_parent / "source", source_parent / "frozen_manifest.json")
    if (job / "frozen_manifest.json").is_file():
        verify(job)
        print("Task13 already prepared and verified; no files modified")
        return
    if (job / "source").exists() or (job / "source.tmp").exists() or (job / "registry.toml").exists():
        raise RuntimeError("Task13 preparation is partially present; inspect before modifying")
    for item in (
        repo / "src/al_pipeline/core/meta_acquisition/tailor.py",
        repo / "src/al_pipeline/core/meta_acquisition/mixed_selection.py",
        repo / "scripts/research/historical_replay/task13/task13_patch.py",
        repo / "scripts/research/historical_replay/task13/task13_worker.sbatch",
    ):
        if not item.is_file():
            raise FileNotFoundError(f"Required version-controlled Task13 overlay is absent: {item}")
    job.mkdir(parents=True, exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    stage = job / "source.tmp"
    try:
        shutil.copytree(source_parent / "source", stage)
        core = stage / "src/al_pipeline/core/meta_acquisition"
        if core.exists():
            raise RuntimeError("Frozen Task11 source unexpectedly already contains Task13 core")
        shutil.copytree(repo / "src/al_pipeline/core/meta_acquisition", core)
        script_dir = stage / "scripts/research/historical_replay/task13"
        for file in sorted((repo / "scripts/research/historical_replay/task13").glob("task13*")):
            if file.suffix in {".py", ".sbatch", ".sh"}:
                shutil.copy2(file, script_dir / file.name)
        patch_source(stage)
        import ast
        for file in stage.rglob("*.py"):
            ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        manifest = relevant_source_hashes(stage)
        source_parent_registry = source_parent / "registry.toml"
        registry = source_parent_registry.read_text(encoding="utf-8")
        if not registry.strip() or "[storage]" not in registry:
            raise ValueError("Frozen parent registry has unexpected format")
        # Do not mutate the parent suite or repository checkout.
        stage.rename(job / "source")
        (job / "registry.toml").write_text(registry, encoding="utf-8")
        write_json(job / "source_parent.json", {
            "suite": TASK11,
            "frozen_manifest_sha256": sha256(source_parent / "frozen_manifest.json"),
            "registry_sha256": sha256(source_parent_registry),
        })
        write_json(job / "scientific_plan.json", scientific_plan())
        write_json(job / "frozen_manifest.json", manifest)
    except Exception:
        # Leave an incomplete source staging tree for manual inspection.
        # Never delete or overwrite a possibly useful failed stage.
        raise
    verify(job)
    print(f"Prepared Task13 frozen source with {len(manifest)} code files: {job}")


def verify(job_dir: str | Path = JOB) -> None:
    job = Path(job_dir)
    if read_json(job / "scientific_plan.json") != scientific_plan():
        raise ValueError("Task13 scientific plan changed")
    verify_source(job / "source", job / "frozen_manifest.json")
    parent = read_json(job / "source_parent.json")
    if parent["suite"] != TASK11:
        raise ValueError("Task13 source-parent suite changed")
    if sha256(TASK11_JOB / "frozen_manifest.json") != parent["frozen_manifest_sha256"]:
        raise ValueError("Frozen Task11 parent manifest changed")
    if sha256(job / "registry.toml") != parent["registry_sha256"]:
        raise ValueError("Task13 registry changed from frozen Task11 registry")
    print(f"Task13 frozen source verified ({len(read_json(job / 'frozen_manifest.json'))} files)")


def runner_args(seed: int, policy: str, *, job_dir: str | Path = JOB) -> list[str]:
    if seed not in SEEDS or policy not in POLICIES:
        raise ValueError("Unexpected Task13 policy or seed")
    job = Path(job_dir)
    return [
        "--registry", str(job / "registry.toml"),
        "--suite-id", SUITE,
        "--bootstrap-source-suite-id", BASE,
        "--require-existing-bootstrap",
        "--training-initialization", "warm_start",
        "--strategies", policy,
        "--seed", str(seed),
        "--split-seed", "135",
        "--initial-fraction", "0.10",
        "--minimum-initial-per-class", "5",
        "--acquisition-fraction", "0.10",
        "--max-cycles", str(CYCLES),
        "--max-epochs", "50",
        "--patience", "10",
        "--batch-size", "512",
        "--validation-batch-size", "2048",
        "--learning-rate", "0.0001",
        "--train-workers", "0",
        "--validation-workers", "0",
        "--inference-workers", "0",
        "--evaluation-workers", "0",
        "--assert-excluded-manifest", str(EXCLUDE),
    ]


def run_runner(
    *, policy: str, seed: int, cycle: int | None, run: bool, job_dir: str | Path = JOB,
) -> None:
    job = Path(job_dir)
    verify(job)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(job / "source/src") + os.pathsep + env.get("PYTHONPATH", "")
    arguments = [sys.executable, str(job / "source/scripts/run_astrominn_closed_loop.py")]
    arguments.extend(runner_args(seed, policy, job_dir=job))
    if run:
        if cycle not in range(1, CYCLES + 1):
            raise ValueError("Task13 training job requires a cycle in 1..4")
        arguments += ["--device", "cuda", "--cycle", str(cycle), "--run"]
    subprocess.run(arguments, env=env, check=True)


def _manifest_object_ids(path: str | Path, column: str) -> set[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {str(row[column]) for row in csv.DictReader(handle)}


def audit_replay_identity(*, job_dir: str | Path = JOB) -> dict:
    """Require Task4-exact L0/U0 identities before Task13 acquisition."""
    from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest
    from al_pipeline.astronomy.data.registry import DatasetRegistry

    job = Path(job_dir)
    dataset = DatasetRegistry.from_toml(job / "registry.toml").resolve_profile("astrominn_ztf").dataset
    old = PERSISTENT / "runs" / TASK4 / "shared/historical_replay_split"
    new = PERSISTENT / "runs" / SUITE / "shared/historical_replay_split"
    if _manifest_object_ids(old / "initial_labeled_manifest.csv", dataset.object_id_column) != _manifest_object_ids(
        new / "initial_labeled_manifest.csv", dataset.object_id_column
    ):
        raise ValueError("Task13 L0 object IDs differ from Task4")
    before = CandidatePoolManifest.from_csv(old / "candidate_pool/candidate_pool.csv")
    after = CandidatePoolManifest.from_csv(new / "candidate_pool/candidate_pool.csv")
    if tuple(before.object_ids) != tuple(after.object_ids):
        raise ValueError("Task13 candidate IDs or order differ from Task4")
    return {
        "l0_objects": len(_manifest_object_ids(new / "initial_labeled_manifest.csv", dataset.object_id_column)),
        "candidate_objects": len(after.object_ids),
        "task4_u0_exact_order_match": True,
        "task4_l0_exact_set_match": True,
        "candidate_sha256": sha256(new / "candidate_pool/candidate_pool.csv"),
    }


def preflight(seed: int = 135, *, job_dir: str | Path = JOB) -> None:
    for policy in POLICIES:
        run_runner(policy=policy, seed=seed, cycle=None, run=False, job_dir=job_dir)
    frozen_python = str(Path(job_dir) / "source/src")
    if frozen_python not in sys.path:
        sys.path.insert(0, frozen_python)
    audit = audit_replay_identity(job_dir=job_dir)
    audit["filesystem_probe"] = filesystem_probe(job_dir=job_dir)
    write_json(Path(job_dir) / "preflight" / f"seed_{seed}.json", audit)
    print(json.dumps(audit, indent=2))


def filesystem_probe(*, job_dir: str | Path = JOB, size_bytes: int = 16 << 20) -> dict:
    """Test checkpoint-filesystem write/fsync/rename with a disposable 16 MiB file.

    This is deliberately isolated in the new Task13 runtime directory, never
    in a prior task's checkpoint directory or any directory containing data.
    """
    if size_bytes <= 0 or size_bytes > (64 << 20):
        raise ValueError("Filesystem probe size must be in (0, 64 MiB]")
    verify(job_dir)
    root = DATA / "runs" / SUITE / "shared" / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix="task13_probe_", suffix=".tmp", dir=root)
    src = Path(filename)
    renamed = src.with_suffix(".checked")
    start = time.perf_counter()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            block = bytes(1 << 20)
            remaining = size_bytes
            while remaining:
                count = min(len(block), remaining)
                handle.write(block[:count])
                remaining -= count
            handle.flush()
            os.fsync(handle.fileno())
        if src.stat().st_size != size_bytes:
            raise OSError("Filesystem probe wrote an unexpected byte count")
        src.rename(renamed)
        with renamed.open("rb") as handle:
            if handle.read(64) != bytes(64):
                raise OSError("Filesystem probe could not read back its first bytes")
        return {
            "passed": True,
            "bytes": size_bytes,
            "write_fsync_rename_seconds": round(time.perf_counter() - start, 3),
        }
    finally:
        # Exactly the two disposable files created by this particular probe.
        for file in (src, renamed):
            if file.is_file():
                file.unlink()


def pilot(policy: str, seed: int = 135, *, job_dir: str | Path = JOB) -> None:
    """Full-model acquisition test; never queries labels, trains, or evaluates."""
    if policy not in POLICIES or seed not in SEEDS:
        raise ValueError("Invalid Task13 pilot policy or seed")
    job = Path(job_dir)
    verify(job)
    audit = audit_replay_identity(job_dir=job)
    completed = job / "pilot" / policy / f"seed_{seed}" / "result.json"
    if completed.exists():
        print(completed.read_text(encoding="utf-8"))
        return

    import numpy as np
    from dataclasses import replace
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import (
        DEFAULT_LEGACY_CODE_DIR,
        predict_astrominn_vit_pool,
    )
    from al_pipeline.astronomy.analysis.task11_labeled_features import predict_confirmed_objects
    from al_pipeline.astronomy.astrominn.training import LABEL_TO_INDEX, MODEL_CLASS_NAMES
    from al_pipeline.core.meta_acquisition.tailor import TailorBandit
    from al_pipeline.core.meta_acquisition.mixed_selection import select_mixed_batch

    registry = DatasetRegistry.from_toml(job / "registry.toml")
    resolved = registry.resolve_profile("astrominn_ztf")
    dataset = resolved.dataset
    split = PERSISTENT / "runs" / SUITE / "shared/historical_replay_split"
    initial = split / "initial_labeled_manifest.csv"
    checkpoint = DATA / "runs" / BASE / "shared" / f"seed_{seed}" / "bootstrap_training/best_vit_tower.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    artifact_dir = Path(resolved.profile.options["checkpoint"]).parent
    counts = np.zeros(len(MODEL_CLASS_NAMES), dtype=np.int64)
    labels_by_id: dict[str, int] = {}
    with initial.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_id = str(row[dataset.object_id_column])
            index = LABEL_TO_INDEX[str(row[dataset.label_column])]
            prior = labels_by_id.setdefault(object_id, index)
            if prior != index:
                raise ValueError("Conflicting labels in historical Task13 L0")
    for index in labels_by_id.values():
        counts[index] += 1
    if len(labels_by_id) != audit["l0_objects"]:
        raise ValueError("L0 confirmed-class counts disagree with preflight")
    start = time.perf_counter()
    candidates = predict_astrominn_vit_pool(
        pool_path=split / "candidate_pool",
        device="cuda",
        batch_size=64,
        num_workers=0,
        legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,
        artifact_dir=artifact_dir,
        vit_checkpoint=checkpoint,
    )
    inference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    anchors = predict_confirmed_objects(
        manifest_path=initial,
        dataset=replace(dataset, train_manifest=initial),
        checkpoint_path=checkpoint,
        artifact_dir=artifact_dir,
        device="cuda",
        batch_size=64,
        num_workers=0,
        legacy_code_dir=DEFAULT_LEGACY_CODE_DIR,
    )
    anchor_seconds = time.perf_counter() - start
    if candidates.embeddings is None or anchors.embeddings is None:
        raise ValueError("Task13 pilot requires classifier-input features")
    if len(candidates.object_ids) != audit["candidate_objects"]:
        raise ValueError("Pilot inference omitted or added candidate objects")
    split_summary = read_json(split / "split_summary.json")
    budget = math.ceil(int(split_summary["historical_objects"]) * 0.10)
    bandit = TailorBandit(class_names=MODEL_CLASS_NAMES)
    allocations = bandit.choose(
        batch_size=budget,
        confirmed_counts=counts,
        method=policy,
        rng=np.random.default_rng(np.random.SeedSequence([seed, 1, 1313])),
    )
    start = time.perf_counter()
    selection = select_mixed_batch(
        object_ids=candidates.object_ids,
        predictions=candidates.predictions,
        embeddings=candidates.embeddings,
        confirmed_counts=counts,
        labeled_probabilities=anchors.predictions.as_probabilities(),
        labeled_embeddings=anchors.embeddings,
        arm_assignments=allocations,
        exploration_rng=np.random.default_rng(np.random.SeedSequence([seed, 1, 1314])),
    )
    selection_seconds = time.perf_counter() - start
    if len(set(selection.ids)) != budget:
        raise RuntimeError("Task13 pilot acquired duplicates or wrong budget")
    if not set(selection.ids).issubset(candidates.object_ids):
        raise RuntimeError("Task13 pilot selected ineligible candidates")
    write_json(completed, {
        "suite": SUITE,
        "policy": policy,
        "seed": seed,
        "cycle": 1,
        "acquisition_budget": budget,
        "selected_ids": list(selection.ids),
        "selection_metadata": selection.metadata,
        "posterior_before": bandit.history_summary(),
        "confirmed_l0_counts": counts.tolist(),
        "candidate_count": len(candidates.object_ids),
        "l0_exact_task4_match": audit["task4_l0_exact_set_match"],
        "u0_exact_task4_match": audit["task4_u0_exact_order_match"],
        "bootstrap_sha256": sha256(checkpoint),
        "selected_unique": True,
        "candidate_inference_seconds": inference_seconds,
        "confirmed_anchor_inference_seconds": anchor_seconds,
        "selection_seconds": selection_seconds,
        "training_performed": False,
        "candidate_hidden_labels_consulted": False,
    })
    print(json.dumps({
        "policy": policy,
        "budget": budget,
        "arm_counts": selection.metadata["arm_counts"],
        "inference_seconds": inference_seconds,
        "anchor_seconds": anchor_seconds,
        "selection_seconds": selection_seconds,
        "training_performed": False,
    }, indent=2))


def check_pilots(*, job_dir: str | Path = JOB) -> dict:
    job = Path(job_dir)
    output = {}
    for policy in POLICIES:
        path = job / "pilot" / policy / "seed_135" / "result.json"
        if not path.is_file():
            raise RuntimeError(f"Task13 acquisition pilot missing: {path}")
        result = read_json(path)
        if (
            result.get("policy") != policy
            or not result.get("selected_unique")
            or result.get("training_performed") is not False
            or result.get("candidate_hidden_labels_consulted") is not False
            or result.get("l0_exact_task4_match") is not True
            or result.get("u0_exact_task4_match") is not True
        ):
            raise RuntimeError(f"Task13 acquisition pilot failed its scientific checks: {policy}")
        if sum(result["selection_metadata"]["arm_counts"].values()) != result["acquisition_budget"]:
            raise RuntimeError("Task13 pilot arm counts do not sum to the acquisition budget")
        output[policy] = result
    if output[POLICIES[0]]["bootstrap_sha256"] != output[POLICIES[1]]["bootstrap_sha256"]:
        raise RuntimeError("Task13 pilot policies used different bootstrap checkpoints")
    return output


def planned_jobs():
    for policy in POLICIES:
        for seed in SEEDS:
            for cycle in range(1, CYCLES + 1):
                yield policy, seed, cycle


def submit(*, walltime: str = "00:45:00", job_dir: str | Path = JOB) -> None:
    """Submit the 24 jobs HELD, retaining an atomic audit after each job."""
    job = Path(job_dir)
    verify(job)
    check_pilots(job_dir=job)
    if not re.fullmatch(r"\d{1,2}:[0-5]\d:[0-5]\d", walltime) or walltime == "00:00:00":
        raise ValueError("Invalid Task13 Slurm walltime")
    ledger = job / "submitted.json"
    if ledger.exists():
        raise RuntimeError("Task13 submission ledger already exists; inspect instead of resubmitting")
    payload: dict[str, object] = {
        "suite": SUITE,
        "status": "submitting_held",
        "walltime": walltime,
        "jobs": [],
    }
    write_json(ledger, payload)
    previous: dict[tuple[str, int], str] = {}
    for policy, seed, cycle in planned_jobs():
        parent = previous.get((policy, seed))
        name = f"T13_{'tail' if policy == 'tailor_diversity' else 'unif'}_{seed}_{cycle}"
        cmd = [
            "sbatch", "--parsable", "--hold", "--time", walltime,
            "--job-name", name,
            "--output", str(job / "logs/%x_%j.out"),
            "--error", str(job / "logs/%x_%j.err"),
        ]
        if parent:
            cmd.extend(["--dependency", f"afterok:{parent}"])
        cmd.extend([
            str(job / "source/scripts/research/historical_replay/task13/task13_worker.sbatch"),
            str(job), "job", policy, str(seed), str(cycle),
        ])
        output = subprocess.check_output(cmd, text=True).strip()
        job_id = output.split(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"Unexpected Slurm job identifier: {output}")
        previous[(policy, seed)] = job_id
        payload["jobs"].append({
            "policy": policy, "seed": seed, "cycle": cycle,
            "job_id": job_id, "parent": parent,
        })
        write_json(ledger, payload)
    if len(payload["jobs"]) != len(POLICIES) * len(SEEDS) * CYCLES:
        raise RuntimeError("Task13 submission does not contain exactly 24 jobs")
    payload["status"] = "held_awaiting_explicit_release"
    write_json(ledger, payload)
    print("24 Task13 jobs submitted HELD, as six 4-cycle afterok chains")
    print("Inspect the ledger and pilots before running `task13.py release`.")


def release(*, job_dir: str | Path = JOB) -> None:
    """Explicitly release all held jobs; safe to retry after a partial release."""
    job = Path(job_dir)
    verify(job)
    check_pilots(job_dir=job)
    ledger = job / "submitted.json"
    payload = read_json(ledger)
    if len(payload.get("jobs", [])) != len(POLICIES) * len(SEEDS) * CYCLES:
        raise RuntimeError("Refusing to release an incomplete Task13 job ledger")
    if payload.get("status") not in {
        "held_awaiting_explicit_release", "partially_released", "released"
    }:
        raise RuntimeError("Task13 ledger is not in a releasable state")
    payload["status"] = "partially_released"
    write_json(ledger, payload)
    for entry in payload["jobs"]:
        job_id = entry["job_id"]
        details = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
        held = "Reason=JobHeldUser" in details or re.search(r"\bPriority=0\b", details)
        if "JobState=PENDING" in details and held:
            subprocess.run(["scontrol", "release", job_id], check=True)
    payload["status"] = "released"
    write_json(ledger, payload)
    print("Task13 release completed. Slurm dependencies still enforce cycle ordering.")


def status(*, job_dir: str | Path = JOB) -> None:
    rows = []
    for policy in POLICIES:
        for seed in SEEDS:
            base = PERSISTENT / "runs" / SUITE / policy / f"seed_{seed}"
            state = base / "run_state.json"
            completed = int(read_json(state).get("last_completed_cycle", 0)) if state.exists() else 0
            rows.append({
                "policy": policy, "seed": seed,
                "last_completed_cycle": completed,
                "complete": completed == CYCLES,
            })
    print(json.dumps({
        "suite": SUITE,
        "planned_training_jobs": len(POLICIES) * len(SEEDS) * CYCLES,
        "completed_series": sum(row["complete"] for row in rows),
        "total_series": len(rows),
        "submission": (
            read_json(Path(job_dir) / "submitted.json").get("status")
            if (Path(job_dir) / "submitted.json").exists() else "not_submitted"
        ),
        "series": rows,
    }, indent=2))


def collect(*, destination: str | Path, job_dir: str | Path = JOB) -> None:
    """Build a compact result archive, excluding model weights and hidden pool labels."""
    job = Path(job_dir)
    verify(job)
    destination = Path(destination)
    if destination.exists() or destination.with_suffix(destination.suffix + ".tmp").exists():
        raise FileExistsError("Choose a new destination; Task13 never overwrites result exports")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths: list[tuple[Path, str]] = []
    for name in ("scientific_plan.json", "frozen_manifest.json", "source_parent.json", "submitted.json"):
        candidate = job / name
        if candidate.is_file():
            paths.append((candidate, "task13_control/" + name))
    for candidate in sorted((job / "pilot").rglob("result.json")):
        paths.append((candidate, "task13_control/" + str(candidate.relative_to(job))))
    for policy in POLICIES:
        for seed in SEEDS:
            path = Path(SUITE) / policy / f"seed_{seed}"
            persistence = PERSISTENT / "runs" / path
            artifacts = DATA / "runs" / path
            for name in ("run_state.json", "experiment.json", "evaluation_history.json"):
                candidate = persistence / name
                if candidate.is_file():
                    paths.append((candidate, "persistent/" + str(path / name)))
            for cycle in range(1, CYCLES + 1):
                rel = f"cycle_{cycle:03d}"
                for name in ("selection.json", "revealed_labels.json", "cycle_summary.json"):
                    candidate = persistence / rel / name
                    if candidate.is_file():
                        paths.append((candidate, "persistent/" + str(path / rel / name)))
            for cycle in range(CYCLES + 1):
                rel = f"cycle_{cycle:03d}/evaluation"
                for name in (
                    "evaluation_summary.json",
                    "validation_per_class_metrics.json",
                    "validation_object_confusion.csv",
                    "validation_sample_confusion.csv",
                ):
                    candidate = artifacts / rel / name
                    if candidate.is_file():
                        paths.append((candidate, "artifacts/" + str(path / rel / name)))
    if not paths:
        raise RuntimeError("Task13 results do not exist yet")
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in paths:
            archive.write(path, arcname=relative)
    temp.replace(destination)
    print(json.dumps({
        "output": str(destination),
        "files": len(paths),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "plan", "prepare", "verify", "preflight", "pilot", "pilot-status",
        "submit", "release", "status", "collect", "filesystem-probe", "job",
    ))
    parser.add_argument("policy", nargs="?", choices=POLICIES)
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("cycle", nargs="?", type=int)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--job-dir", type=Path, default=JOB)
    parser.add_argument("--walltime", default="00:45:00")
    parser.add_argument("--pilot-seed", type=int, default=135)
    parser.add_argument("--pilot-policy", choices=POLICIES)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(scientific_plan(), indent=2))
    elif args.command == "prepare":
        prepare(repo=args.repo, job_dir=args.job_dir)
    elif args.command == "verify":
        verify(args.job_dir)
    elif args.command == "preflight":
        preflight(seed=args.pilot_seed, job_dir=args.job_dir)
    elif args.command == "filesystem-probe":
        print(json.dumps(filesystem_probe(job_dir=args.job_dir), indent=2))
    elif args.command == "pilot":
        pilot(args.pilot_policy or args.policy, args.pilot_seed, job_dir=args.job_dir)
    elif args.command == "pilot-status":
        print(json.dumps({
            policy: {
                "found": (Path(args.job_dir) / "pilot" / policy / "seed_135/result.json").is_file(),
                "path": str(Path(args.job_dir) / "pilot" / policy / "seed_135/result.json"),
            } for policy in POLICIES
        }, indent=2))
    elif args.command == "submit":
        submit(walltime=args.walltime, job_dir=args.job_dir)
    elif args.command == "release":
        release(job_dir=args.job_dir)
    elif args.command == "status":
        status(job_dir=args.job_dir)
    elif args.command == "collect":
        if args.destination is None:
            raise ValueError("collect requires --destination PATH")
        collect(destination=args.destination, job_dir=args.job_dir)
    else:
        if args.policy not in POLICIES or args.seed not in SEEDS or args.cycle not in range(1, CYCLES + 1):
            raise ValueError("Task13 job requires a valid policy, seed and cycle 1..4")
        run_runner(policy=args.policy, seed=args.seed, cycle=args.cycle, run=True, job_dir=args.job_dir)


if __name__ == "__main__":
    main()
