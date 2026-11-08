#!/usr/bin/env python3
"""Build Task 6B's independent runner and isolated training implementation.

The production repository's training.py is NOT modified. Its science change is
applied only to the source snapshot. All replacements fail closed if upstream
changes invalidate the audited anchors.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

SUITE = 'astrominn_closed_loop_warm_f1best_085_v1'


def replace_one(text: str, old: str, new: str) -> str:
    count=text.count(old)
    if count != 1:
        raise ValueError(f'Expected one upstream anchor, got {count}: {old[:130]!r}')
    return text.replace(old,new,1)


def build_runner(repo: Path) -> str:
    src=repo/'scripts/research/historical_replay/task4/task4_build_runner.py'
    spec=importlib.util.spec_from_file_location('task6b_upstream_builder',src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load Task4 builder: {src}')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result=mod.build_runner((repo/'scripts/run_astrominn_historical_replay.py').read_text())
    result=replace_one(result,
        '    if args.bootstrap_source_suite_id == args.suite_id:\n'
        '        raise ValueError("Use a distinct suite ID for Task 4.")',
        '    if args.bootstrap_source_suite_id != args.suite_id:\n'
        '        raise ValueError("Task6B requires its OWN L0 training")')
    result=replace_one(result,
        '    if not args.require_existing_bootstrap:\n'
        '        raise ValueError("Task 4 must reuse an existing, validated L0 bootstrap.")',
        '    if args.run and not args.bootstrap_only and not args.require_existing_bootstrap:\n'
        '        raise ValueError("Task6B cycle jobs require their completed own L0 bootstrap")')
    result=replace_one(result,
        '        / args.bootstrap_source_suite_id\n        / "shared"\n'
        '        / f"seed_{args.seed}"\n        / "bootstrap_training"',
        '        / args.suite_id\n        / "shared"\n'
        '        / f"seed_{args.seed}"\n        / "bootstrap_training"')
    result=replace_one(result,
        '        run_id=f"{args.bootstrap_source_suite_id}_bootstrap_seed{args.seed}",',
        '        run_id=f"{args.suite_id}_bootstrap_seed{args.seed}",')
    result=replace_one(result,
        '    policy = AstroMiNNViTTrainingConfig(\n',
        '    policy = AstroMiNNViTTrainingConfig(\n'
        '        class_weight_exponent=0.85,\n'
        '        checkpoint_selection="object_macro_f1_best",\n')
    result=replace_one(result,
        '        "cycle_training_initialization": args.training_initialization,',
        '        "cycle_training_initialization": args.training_initialization,\n'
        '        "task6b_bootstrap_weight_exponent": 0.85,\n'
        '        "task6b_cycle_weight_exponent": 0.85,\n'
        '        "task6b_checkpoint_selection": "object_macro_f1_best",\n'
        '        "task6b_early_stopping": "best_weighted_validation_loss",\n'
        '        "task6b_object_weighting": "per_sample",')
    result=replace_one(result,
        '    if args.bootstrap_only and not args.run:\n',
        f'    if args.suite_id != "{SUITE}":\n'
        '        raise ValueError("Task6B requires its own dedicated suite ID")\n'
        '    if args.seed not in (135, 246, 357):\n'
        '        raise ValueError("Unplanned Task6B seed")\n'
        '    if args.strategies not in (["entropy"], ["entropy_balanced_explore"]):\n'
        '        raise ValueError("Unplanned Task6B acquisition policy")\n'
        '    if args.bootstrap_only and args.require_existing_bootstrap:\n'
        '        raise ValueError("L0 training must not require a bootstrap")\n'
        '    if args.bootstrap_only and not args.run:\n')
    result=replace_one(result,
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))',
        '    if (policy.class_weight_exponent != 0.85 or\n'
        '            policy.checkpoint_selection != "object_macro_f1_best" or\n'
        '            policy.class_weight_source != "training_manifest" or\n'
        '            policy.new_object_weighting != "per_sample"):\n'
        '        raise AssertionError("Task6B scientific policy changed")\n'
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))')
    # Task4's inherited human-readable field must describe the actual checkpoint.
    result=replace_one(result, 'L0_then_previous_loss_best_checkpoint',
                       'L0_then_previous_object_macro_f1_best_checkpoint')
    if result.count('checkpoint_selection="object_macro_f1_best"')!=1:
        raise AssertionError('Missing exclusive checkpoint-selection configuration')
    return result


def patch_training(source: str) -> str:
    """Switch selected checkpoint to *object* macro-F1 without altering optimization.

    Weighted validation loss remains the learning-rate/early-stop criterion.
    The loss-best state is retained in a separate file for within-fit analysis.
    Object probabilities are represented by the latest alert only, matching
    the experiment evaluator and acquisition definitions.
    """
    s=source
    s=replace_one(s,'from sklearn.metrics import auc, precision_recall_curve',
                  'from sklearn.metrics import auc, precision_recall_curve, f1_score')
    s=replace_one(s, '    class_weight_source: ClassWeightSource = "legacy_reference"',
                  '    class_weight_source: ClassWeightSource = "legacy_reference"\n'
                  '    checkpoint_selection: Literal["loss_best", "object_macro_f1_best"] = "loss_best"')
    # Insert validation just before the next class method, after __post_init__.
    s=replace_one(s, '\n    @classmethod\n    def legacy_seed135(',
                  '\n    # Task 6B uses the *same* optimizer and validation loss; only selection changes.\n'
                  '    def _validate_checkpoint_selection(self) -> None:\n'
                  '        if self.checkpoint_selection not in ("loss_best", "object_macro_f1_best"):\n'
                  '            raise ValueError("Invalid checkpoint selection mode")\n\n'
                  '    @classmethod\n    def legacy_seed135(')
    # Config method above is explicit; make __post_init__ call it.
    s=replace_one(s,
        '            raise ValueError(\n'
        '                "class_weight_source must be \'legacy_reference\' or \'training_manifest\'."\n'
        '            )',
        '            raise ValueError(\n'
        '                "class_weight_source must be \'legacy_reference\' or \'training_manifest\'."\n'
        '            )\n'
        '        self._validate_checkpoint_selection()')
    s=replace_one(s,
        '        checkpoint_path = request.output_dir / "best_vit_tower.pth"',
        '        checkpoint_path = request.output_dir / "best_vit_tower.pth"\n'
        '        loss_checkpoint_path = request.output_dir / "best_loss_vit_tower.pth"')
    s=replace_one(s,
        '        best_validation_loss = float("inf")\n'
        '        best_epoch = 0',
        '        best_validation_loss = float("inf")\n'
        '        best_object_macro_f1 = float("-inf")\n'
        '        selected_validation_loss = float("inf")\n'
        '        best_loss_epoch = 0\n'
        '        best_epoch = 0')
    s=replace_one(s,
        '            validation_loss = _evaluate_loss(\n'
        '                loader=validation_loader,\n'
        '                model=model,\n'
        '                criterion=criterion,\n'
        '                device=self.device,\n'
        '            )',
        '            validation_loss, object_macro_f1 = _task6b_evaluate_loss_and_object_f1(\n'
        '                loader=validation_loader, model=model, criterion=criterion,\n'
        '                device=self.device,\n'
        '            )')
    old='''            improved = validation_loss < best_validation_loss

            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                _torch_save_atomic(
                    model.vit_tower.state_dict(),
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
'''
    new='''            # Preserve loss-based optimization, LR scheduler and patience.
            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                best_loss_epoch = epoch
                epochs_without_improvement = 0
                _torch_save_atomic(model.vit_tower.state_dict(), loss_checkpoint_path)
            else:
                epochs_without_improvement += 1

            # Select the highest macro-F1 over the *same* epochs considered
            # under loss-based early stopping. Ties: lower loss, then earlier.
            f1_improved = (object_macro_f1 > best_object_macro_f1 + 1e-12 or
                           (abs(object_macro_f1 - best_object_macro_f1) <= 1e-12 and
                            validation_loss < selected_validation_loss - 1e-12))
            if f1_improved:
                best_object_macro_f1 = object_macro_f1
                selected_validation_loss = validation_loss
            select_current = (f1_improved if self.policy.checkpoint_selection ==
                              "object_macro_f1_best" else improved)
            if select_current:
                best_epoch = epoch
                _torch_save_atomic(model.vit_tower.state_dict(), checkpoint_path)
'''
    s=replace_one(s,old,new)
    s=replace_one(s,
        '                "validation_loss": validation_loss,',
        '                "validation_loss": validation_loss,\n'
        '                "validation_object_macro_f1": object_macro_f1,\n'
        '                "best_validation_object_macro_f1": best_object_macro_f1,\n'
        '                "best_loss_epoch": best_loss_epoch,\n'
        '                "selected_checkpoint_improved": select_current,')
    s=replace_one(s,
        '                    "best_epoch": best_epoch,',
        '                    "best_epoch": best_epoch,\n'
        '                    "best_loss_epoch": best_loss_epoch,\n'
        '                    "best_object_macro_f1": best_object_macro_f1,')
    s=replace_one(s,
        '            "best_epoch": best_epoch,\n'
        '            "epochs_completed": len(history),',
        '            "best_epoch": best_epoch,\n'
        '            "best_loss_epoch": best_loss_epoch,\n'
        '            "best_object_macro_f1": best_object_macro_f1,\n'
        '            "selected_checkpoint_validation_loss": selected_validation_loss,\n'
        '            "checkpoint_selection": self.policy.checkpoint_selection,\n'
        '            "loss_best_checkpoint_path": str(loss_checkpoint_path),\n'
        '            "epochs_completed": len(history),')
    s=replace_one(s,
        '        result = TrainingResult(\n            checkpoint_path=Path(summary["checkpoint_path"]),',
        '        if summary.get("checkpoint_selection") != self.policy.checkpoint_selection:\n'
        '            raise ValueError("Task6B checkpoint-selection provenance mismatch")\n'
        '        loss_best = Path(summary.get("loss_best_checkpoint_path", ""))\n'
        '        if not loss_best.is_file():\n'
        '            raise FileNotFoundError(f"Missing saved loss-best checkpoint: {loss_best}")\n'
        '        result = TrainingResult(\n            checkpoint_path=Path(summary["checkpoint_path"]),')
    s=replace_one(s,
        'def _evaluate_auprc(\n',
        '''def _task6b_evaluate_loss_and_object_f1(
    *, loader: DataLoader, model: nn.Module,
    criterion: nn.Module, device: torch.device,
) -> tuple[float, float]:
    """Validation loss plus one prediction per object (latest alert).

    Both quantities share the SAME inference pass, avoiding an additional
    GPU forward pass per epoch. All five classes contribute to macro-F1.
    """
    model.eval()
    total_loss = 0.0
    # obj_id -> (latest_alert, true_label, predicted_label)
    latest: dict[str, tuple[int, int, int]] = {}
    with torch.inference_mode():
        for batch in loader:
            metadata = batch["metadata"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            logits = _extract_logits(model(metadata, image=image))
            per_sample_loss = criterion(logits, target)
            total_loss += float(per_sample_loss.mean().detach().cpu())
            predictions = logits.argmax(dim=1).detach().cpu().tolist()
            targets = batch["target_index"]
            if isinstance(targets, torch.Tensor):
                targets = targets.tolist()
            alerts = batch["alert"]
            if isinstance(alerts, torch.Tensor):
                alerts = alerts.tolist()
            for oid, alert, truth, pred in zip(
                batch["obj_id"], alerts, targets, predictions, strict=True,
            ):
                key = str(oid)
                record = (int(alert), int(truth), int(pred))
                prior = latest.get(key)
                if prior is not None and prior[1] != record[1]:
                    raise ValueError(f"Conflicting validation labels for object {key}")
                if prior is not None and prior[0] == record[0]:
                    raise ValueError(f"Duplicate latest-alert entry for {key}")
                if prior is None or record[0] > prior[0]:
                    latest[key] = record
    if not latest or len(loader) == 0:
        raise ValueError("Empty validation loader")
    truths = [row[1] for row in latest.values()]
    preds = [row[2] for row in latest.values()]
    score = float(f1_score(
        truths, preds, labels=list(range(len(MODEL_CLASS_NAMES))),
        average="macro", zero_division=0,
    ))
    return total_loss / len(loader), score


def _evaluate_auprc(
''')
    if 'object_macro_f1_best' not in s or '_task6b_evaluate_loss_and_object_f1' not in s:
        raise AssertionError('Missing Task6B selection code')
    return s


def build_snapshot_files(repo:Path) -> tuple[str,str]:
    return build_runner(repo), patch_training((repo/'src/al_pipeline/astronomy/astrominn/training.py').read_text())
