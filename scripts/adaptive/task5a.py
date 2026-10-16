#!/usr/bin/env python3
"""Task 5A CPU representation diagnostics: prepare, extract shards, report.

Runs completely outside Task 4 and Experiment B output trees. No training of
AstroMiNN or acquisition; only a balanced linear probe trained on original L0.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from al_pipeline.astronomy.representation.core import (
    REPRESENTATIONS, SPLITS, describe_rows, digest_file, digest_object, fit_and_evaluate,
    latest_rows, shard_for, stratified_subset, validate_disjoint, write_json_atomic,
    write_npz_atomic,
)

BASE_SUITE = "astrominn_historical_replay_v1"
PROFILE = "astrominn_ztf"


def _paths(args: argparse.Namespace):
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    registry = DatasetRegistry.from_toml(args.registry)
    resolved = registry.resolve_profile(args.profile)
    dataset = resolved.dataset
    if dataset.object_id_column is None or dataset.label_column is None:
        raise ValueError("Dataset registry must specify object and label columns")
    split_root = resolved.storage.persistent_root / "runs" / BASE_SUITE / "shared" / "historical_replay_split"
    manifests = {
        "train": split_root / "initial_labeled_manifest.csv",
        "validation": dataset.validation_manifest,
        "test": dataset.test_manifest,
    }
    if any(value is None for value in manifests.values()):
        raise ValueError("The original historical validation and test manifests are required")
    checkpoint = resolved.storage.data_root / "runs" / BASE_SUITE / "shared" / f"seed_{args.seed}" / "bootstrap_training" / "best_vit_tower.pth"
    published = resolved.profile.options.get("checkpoint")
    if published is None:
        raise ValueError("AstroMiNN profile has no published checkpoint")
    return resolved, manifests, checkpoint, Path(published).parent


def prepare(args: argparse.Namespace) -> None:
    from al_pipeline.astronomy.astrominn.training import LABEL_TO_INDEX, MODEL_CLASS_NAMES
    resolved, manifests, checkpoint, artifact_dir = _paths(args)
    if args.mode == "smoke" and args.shards != 1:
        raise ValueError("Use exactly 1 shard in smoke mode")
    if args.mode == "full" and args.cap_per_class is not None:
        raise ValueError("Full mode must use all objects, not a capped subset")
    if len(set(args.representations)) != len(args.representations):
        raise ValueError("Duplicate representations")
    for name in args.representations:
        if name not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {name}")
    required_artifacts = [artifact_dir / "astroMiNN_config.json",
                          artifact_dir / "best_lc_tower_135.pth",
                          artifact_dir / "best_spatial_tower_135.pth",
                          artifact_dir / "best_coord_tower_135.pth"]
    inputs = [args.registry, checkpoint, *required_artifacts, *manifests.values()]
    for item in inputs:
        if not Path(item).is_file():
            raise FileNotFoundError(item)
    full_rows = {split: latest_rows(Path(path), resolved.dataset.object_id_column,
                              resolved.dataset.label_column, LABEL_TO_INDEX)
                 for split, path in manifests.items()}
    # Check leakage in the COMPLETE source splits, before smoke subsampling.
    validate_disjoint(full_rows)
    cap = (3 if args.cap_per_class is None else args.cap_per_class) if args.mode == "smoke" else None
    rows = {split: stratified_subset(data, cap) for split, data in full_rows.items()}
    if set(item["target"] for item in rows["train"]) != set(range(len(MODEL_CLASS_NAMES))):
        raise ValueError("Original L0 does not contain all five model classes")
    if args.shards <= 0:
        raise ValueError("Invalid shard count")
    from al_pipeline.astronomy.representation import core as core_module
    from al_pipeline.astronomy.representation import astrominn as model_module
    code_files = [Path(__file__).resolve(), Path(core_module.__file__).resolve(),
                  Path(model_module.__file__).resolve()]
    plan = {
        "schema": "task5a_representation_v1",
        "code_hashes": {str(p): digest_file(p) for p in code_files},
        "mode": args.mode, "seed": args.seed, "shards": args.shards,
        "profile": args.profile, "registry": str(args.registry.resolve()),
        "registry_sha256": digest_file(args.registry),
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest_file(checkpoint),
        "artifact_dir": str(artifact_dir),
        "artifact_hashes": {p.name: digest_file(p) for p in required_artifacts},
        "legacy_code_dir": str(args.legacy_code_dir),
        "representations": list(args.representations), "class_names": list(MODEL_CLASS_NAMES),
        "object_column": resolved.dataset.object_id_column,
        "label_column": resolved.dataset.label_column,
        "splits": {name: {
            "manifest": str(manifests[name]), "sha256": digest_file(manifests[name]),
            "objects": data, "counts": describe_rows(data),
        } for name, data in rows.items()},
        "probe": {"C": 1.0, "class_weight": "balanced", "split_for_fit": "train"},
        "restrictions": {"train_source": "original_L0_only", "hidden_U0_labels_read": False,
                         "validation_test_training": False, "task4_files_modified": False},
    }
    plan["plan_sha256"] = digest_object(plan)
    destination = args.run_dir / "plan.json"
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError(f"Immutable 5A plan has changed: {destination}. Use a new run dir.")
        print(f"5A prepared plan already verified: {destination}")
        return
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError(f"Nonempty 5A run directory without plan: {args.run_dir}")
    write_json_atomic(destination, plan)
    print(f"Task 5A {args.mode} plan: {destination}")
    for split in SPLITS:
        print(f"  {split}: {plan['splits'][split]['counts']}")
    print("No jobs submitted. Original Experiment B and Task 4 untouched.")


def load_plan(run_dir: Path, *, verify_files: bool) -> dict:
    plan_path = run_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    sha = plan.pop("plan_sha256")
    if digest_object(plan) != sha or plan["schema"] != "task5a_representation_v1":
        raise ValueError("Plan hash/schema verification failed")
    plan["plan_sha256"] = sha
    if verify_files:
        manifest_sources = {"registry": (plan["registry"], plan["registry_sha256"]),
                            "checkpoint": (plan["checkpoint"], plan["checkpoint_sha256"])}
        manifest_sources.update({name: (entry["manifest"], entry["sha256"])
                                 for name, entry in plan["splits"].items()})
        manifest_sources.update({name: (str(Path(plan["artifact_dir"]) / name), digest)
                                 for name, digest in plan["artifact_hashes"].items()})
        manifest_sources.update({"code:" + name: (name, digest)
                                 for name, digest in plan["code_hashes"].items()})
        for name, (path, expected) in manifest_sources.items():
            if digest_file(path) != expected:
                raise ValueError(f"Source changed since 5A prepare: {name}: {path}")
    return plan


def extract(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    from al_pipeline.astronomy.astrominn.training import AstroMiNNViTManifestDataset
    from al_pipeline.astronomy.astrominn.inference.astrominn_vit import load_astrominn_vit_model
    from al_pipeline.astronomy.representation.astrominn import extract_features

    plan = load_plan(args.run_dir, verify_files=True)
    if args.split not in SPLITS or not 0 <= args.shard < plan["shards"]:
        raise ValueError("Invalid split/shard")
    if plan["mode"] == "full" and args.max_objects is not None:
        raise ValueError("Object limits are forbidden in full mode")
    dest = args.run_dir / "features" / args.split / f"shard_{args.shard:03d}_of_{plan['shards']:03d}.npz"
    metadata = dest.with_suffix(".json")
    selected = [item for item in plan["splits"][args.split]["objects"]
                if shard_for(item["object_id"], plan["shards"]) == args.shard]
    if args.max_objects is not None:
        if plan["mode"] != "smoke":
            raise ValueError("max_objects only allowed for smoke")
        selected = selected[:args.max_objects]
    if dest.exists() and metadata.exists():
        current = json.loads(metadata.read_text(encoding="utf-8"))
        if current.get("plan_sha256") != plan["plan_sha256"] or current.get("sha256") != digest_file(dest):
            raise ValueError(f"Cached features have invalid provenance: {dest}")
        print(f"Verified existing shard, skipped: {dest}")
        return
    if dest.exists() and not metadata.exists():
        # Interrupted after atomic NPZ write but before the commit metadata.
        # Such a shard was never declared complete, so safely re-extract it.
        print(f"Recovering uncommitted feature shard: {dest}")
        dest.unlink()
    if metadata.exists() and not dest.exists():
        raise ValueError(f"Shard metadata exists without data; inspect: {metadata}")
    if not selected:
        arrays = {"object_ids": np.asarray([], dtype=str), "labels": np.asarray([], dtype=np.int64),
                  "alerts": np.asarray([], dtype=np.int64),
                  **{name: np.empty((0, 0), dtype=np.float32) for name in plan["representations"]}}
    else:
        registry = DatasetRegistry.from_toml(plan["registry"])
        dataset = registry.resolve_profile(plan["profile"]).dataset
        ds = AstroMiNNViTManifestDataset(
            manifest_path=plan["splits"][args.split]["manifest"], dataset=dataset,
            artifact_dir=plan["artifact_dir"], augment=False, cache_in_memory=False)
        subset = Subset(ds, [int(row["row"]) for row in selected])
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        torch.set_num_threads(args.threads)
        model = load_astrominn_vit_model(
            artifact_dir=plan["artifact_dir"], legacy_code_dir=plan["legacy_code_dir"],
            vit_checkpoint=plan["checkpoint"])
        arrays = extract_features(model, loader, representations=plan["representations"], device="cpu")
        for i, item in enumerate(selected):
            if (arrays["object_ids"][i] != item["object_id"]
                    or arrays["labels"][i] != item["target"]
                    or arrays["alerts"][i] != item["alert"]):
                raise ValueError(f"Provenance mismatch in extraction at object {item['object_id']}")
        del model
    write_npz_atomic(dest, **arrays)
    write_json_atomic(metadata, {"schema": "task5a_feature_shard_v1", "plan_sha256": plan["plan_sha256"],
                                 "sha256": digest_file(dest), "split": args.split,
                                 "shard": args.shard, "objects": len(selected),
                                 "feature_dimensions": {n: int(arrays[n].shape[1]) for n in plan["representations"]}})
    print(f"Wrote {len(selected)} objects: {dest}")


def _load_split(run_dir: Path, plan: dict, split: str) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {
        key: [] for key in ("object_ids", "labels", "alerts", *plan["representations"])}
    for shard in range(plan["shards"]):
        path = run_dir / "features" / split / f"shard_{shard:03d}_of_{plan['shards']:03d}.npz"
        metadata = path.with_suffix(".json")
        if not path.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"Missing 5A feature shard: {path}")
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("schema") != "task5a_feature_shard_v1" or record.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError(f"Mismatched feature shard provenance: {path}")
        if digest_file(path) != record.get("sha256"):
            raise ValueError(f"Corrupt feature shard: {path}")
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(outputs):
                raise ValueError(f"Wrong feature arrays in {path}")
            for name in outputs:
                outputs[name].append(np.array(saved[name]))
    rows = plan["splits"][split]["objects"]
    merged = {}
    for name, arrays in outputs.items():
        nonempty = [arr for arr in arrays if len(arr)]
        if not nonempty:
            raise ValueError(f"No objects extracted for {split}")
        merged[name] = np.concatenate(nonempty, axis=0)
    observed = {obj: (int(y), int(alert)) for obj, y, alert in zip(
        merged["object_ids"], merged["labels"], merged["alerts"], strict=True)}
    expected = {row["object_id"]: (row["target"], row["alert"]) for row in rows}
    if len(observed) != len(merged["object_ids"]) or observed != expected:
        raise ValueError(f"Missing, duplicate, or mislabeled objects for {split}")
    for name in plan["representations"]:
        arr = merged[name]
        if arr.ndim != 2 or len(arr) != len(observed) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid {name} features in {split}")
    return merged


def report(args: argparse.Namespace) -> None:
    plan = load_plan(args.run_dir, verify_files=True)
    splits = {name: _load_split(args.run_dir, plan, name) for name in SPLITS}
    results = {
        "schema": "task5a_probe_report_v1", "plan_sha256": plan["plan_sha256"],
        "mode": plan["mode"], "seed": plan["seed"],
        "limitations": ["Probes fit exclusively on L0", "Validation and test are report-only",
                        "No hidden U0 labels used", "Representation diagnostics do not establish closed-loop AL gains"],
        "representations": {},
    }
    for name in plan["representations"]:
        fit = fit_and_evaluate(
            splits["train"][name], splits["train"]["labels"],
            {split: (splits[split][name], splits[split]["labels"]) for split in ("validation", "test")},
            names=tuple(plan["class_names"]), c=plan["probe"]["C"])
        results["representations"][name] = fit
    outfile = args.run_dir / "report.json"
    if outfile.exists():
        previous = json.loads(outfile.read_text(encoding="utf-8"))
        if previous != results:
            raise ValueError(f"Previously written report differs: {outfile}")
    else:
        write_json_atomic(outfile, results)
    table = args.run_dir / "metrics.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        fields = ["representation", "split", "objects", "accuracy", "balanced_accuracy",
                  "macro_f1", "log_loss", "nearest_train_object_label_agreement"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for representation, result in results["representations"].items():
            for split, stats in result["splits"].items():
                writer.writerow({"representation": representation, "split": split,
                                 **{key: stats[key] for key in fields[2:]}})
    per_class_path = args.run_dir / "per_class_metrics.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["representation", "split", "class", "support", "precision", "recall", "f1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for representation, result in results["representations"].items():
            for split, stats in result["splits"].items():
                for class_name, metrics in stats["per_class"].items():
                    writer.writerow({"representation": representation, "split": split,
                                     "class": class_name, **metrics})
    # PCA is fitted to L0 only. Plotting validation geometry is exploratory;
    # no PCA axis, hyperparameter, or report result enters acquisition.
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    projections = args.run_dir / "pca_projections.csv"
    with projections.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["representation", "split", "object_id", "class_name", "pc1", "pc2"])
        for representation in plan["representations"]:
            scaler = StandardScaler().fit(splits["train"][representation])
            train_matrix = scaler.transform(splits["train"][representation])
            if min(train_matrix.shape) < 2:
                continue
            pca = PCA(n_components=2, random_state=135).fit(train_matrix)
            for split in SPLITS:
                transformed = pca.transform(scaler.transform(splits[split][representation]))
                for object_id, target, coord in zip(
                    splits[split]["object_ids"], splits[split]["labels"], transformed, strict=True
                ):
                    writer.writerow([representation, split, str(object_id),
                                     plan["class_names"][int(target)], float(coord[0]), float(coord[1])])
    print(f"PCA coordinates (train-fit only): {projections}")
    lines = [f"# Task 5A — {plan['mode']} representation diagnostics", "",
             f"Bootstrap seed: {plan['seed']}; fit: original L0 only.", "",
             "| Representation | Validation macro-F1 | Validation balanced accuracy | Test macro-F1 |",
             "|---|---:|---:|---:|"]
    for name, obj in results["representations"].items():
        val, test = obj["splits"]["validation"], obj["splits"]["test"]
        lines.append(f"| {name} | {val['macro_f1']:.4f} | {val['balanced_accuracy']:.4f} | {test['macro_f1']:.4f} |")
    lines += ["", "Test outcomes are descriptive only and must not be used for model selection.",
              "The smoke run is an engineering check, not scientifically meaningful evidence."]
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Outputs: {outfile}, {table}, {per_class_path}, {args.run_dir / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--registry", type=Path, required=True)
    prepare_parser.add_argument("--profile", default=PROFILE)
    prepare_parser.add_argument("--seed", type=int, default=135)
    prepare_parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    prepare_parser.add_argument("--shards", type=int, default=1)
    prepare_parser.add_argument("--cap-per-class", type=int, default=None)
    prepare_parser.add_argument("--legacy-code-dir", type=Path,
                                default=Path("/projects/bcrv/ffontinelenunes/things_for_hailey"))
    prepare_parser.add_argument("--representations", nargs="+", default=["vit_cls", "expert_concat"],
                                choices=REPRESENTATIONS)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--split", required=True, choices=SPLITS)
    extraction.add_argument("--shard", type=int, required=True)
    extraction.add_argument("--batch-size", type=int, default=8)
    extraction.add_argument("--threads", type=int, default=4)
    extraction.add_argument("--max-objects", type=int)
    commands.add_parser("report")
    commands.add_parser("verify")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "extract":
        if args.batch_size < 1 or args.threads < 1:
            raise ValueError("batch-size and threads must be positive")
        extract(args)
    elif args.action == "report":
        report(args)
    elif args.action == "verify":
        plan = load_plan(args.run_dir, verify_files=True)
        print(f"5A {plan['mode']} plan verified: {plan['plan_sha256']}")


if __name__ == "__main__":
    main()
