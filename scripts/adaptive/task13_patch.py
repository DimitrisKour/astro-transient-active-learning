#!/usr/bin/env python3
"""Fail-closed Task13 overlay applied ONLY to a fresh frozen Task11 copy.

Never edits the Task11 source itself or an active Git checkout. Every anchor
must match the byte-for-byte verified Task11 snapshot provided on Delta.
"""

from __future__ import annotations

from pathlib import Path


META_POLICIES = ("tailor_diversity", "uniform_mixture")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    occurrences = source.count(old)
    if occurrences != 1:
        raise ValueError(
            f"Task13 frozen-source anchor mismatch [{label}]: "
            f"expected one occurrence, found {occurrences}"
        )
    return source.replace(old, new, 1)


def patch_source(root: str | Path) -> list[str]:
    """Patch a staged source tree, failing before runtime if any anchor moved."""
    root = Path(root)
    paths = {
        "config": "src/al_pipeline/core/experiments/config.py",
        "astrominn": "src/al_pipeline/astronomy/experiments/astrominn.py",
        "simulated": "src/al_pipeline/astronomy/experiments/simulated.py",
        "runner": "scripts/run_astrominn_closed_loop.py",
    }
    for relative in paths.values():
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)
    if not (root / "src/al_pipeline/core/meta_acquisition/mixed_selection.py").is_file():
        raise FileNotFoundError("Task13 generic core modules must be copied before patching")

    path = root / paths["config"]
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        'StrategyName: TypeAlias = Literal[\n',
        'StrategyName: TypeAlias = Literal[\n    "tailor_diversity",\n    "uniform_mixture",\n',
        "strategy-literal",
    )
    source = replace_once(
        source,
        '        if self.strategy not in {\n',
        '        if self.strategy not in {\n            "tailor_diversity", "uniform_mixture",\n',
        "strategy-validation",
    )
    path.write_text(source, encoding="utf-8")

    path = root / paths["astrominn"]
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        '            "badge", "core_set", "fisher_batch",\n        }',
        '            "badge", "core_set", "fisher_batch",\n'
        '            "tailor_diversity", "uniform_mixture",\n        }',
        "suite-policy-validation",
    )
    path.write_text(source, encoding="utf-8")

    path = root / paths["runner"]
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        '            "badge", "core_set", "fisher_batch",\n        ),',
        '            "badge", "core_set", "fisher_batch",\n'
        '            "tailor_diversity", "uniform_mixture",\n        ),',
        "runner-strategy-choices",
    )
    source = replace_once(
        source,
        '    if set(args.strategies) - {"badge", "core_set", "fisher_batch"}:\n'
        '        raise ValueError("Task11 runner permits only the three frozen Task11 acquisition policies")\n',
        '    if set(args.strategies) - {"tailor_diversity", "uniform_mixture"}:\n'
        '        raise ValueError("Task13 runner permits only the two frozen meta-acquisition policies")\n',
        "runner-safety-guard",
    )
    path.write_text(source, encoding="utf-8")

    path = root / paths["simulated"]
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        'from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector\n',
        'from al_pipeline.core.batch_selection import DiversitySelector, ExplorationSelector\n'
        'from al_pipeline.core.meta_acquisition.tailor import TailorBandit, TailorSettings\n'
        'from al_pipeline.core.meta_acquisition.mixed_selection import select_mixed_batch\n',
        "meta-imports",
    )
    source = replace_once(
        source,
        '        if config.strategy == "entropy_balanced_explore":\n',
        '        if config.strategy in {"entropy_balanced_explore", "tailor_diversity", "uniform_mixture"}:\n',
        "meta-requires-class-names",
    )
    source = replace_once(
        source,
        '    ) -> SelectionBatch:\n        if path.exists():\n            selection = _load_selection(path)\n',
        '''    ) -> SelectionBatch:
        if self.config.strategy in {"tailor_diversity", "uniform_mixture"}:
            if path.exists():
                selection = _load_selection(path)
            else:
                selection = self._task13_meta_select(
                    inference=inference, budget=budget, cycle=cycle
                )
                _save_selection(selection, path)
            if len(selection) != budget:
                raise ValueError("Task13 persisted selection has a different acquisition budget")
            if not set(selection.ids).issubset(inference.object_ids):
                raise ValueError("Task13 selection contains an ineligible object")
            return selection
        if path.exists():
            selection = _load_selection(path)
''',
        "simulation-meta-dispatch",
    )
    source = replace_once(
        source,
        '    def _labels_or_resume(\n',
        '''    def _task13_meta_select(
        self, *, inference: ClassificationInferenceResult, budget: int, cycle: int
    ) -> SelectionBatch:
        """Reconstruct bandit only from PREVIOUSLY REVEALED object labels.

        The current cycle's hidden candidate labels are never read here.
        Replaying durable history avoids any independently mutable posterior
        that could be advanced twice after an interrupted job.
        """
        settings = TailorSettings()
        bandit = TailorBandit(class_names=self.class_names, settings=settings)
        for previous in range(1, cycle):
            previous_dir = self.run_dir / f"cycle_{previous:03d}"
            selection_path = previous_dir / "selection.json"
            labels_path = previous_dir / "revealed_labels.json"
            if not selection_path.is_file() or not labels_path.is_file():
                raise RuntimeError(f"Missing previously completed Task13 cycle {previous}")
            selected = _load_selection(selection_path)
            revealed = _load_labeled(labels_path)
            if tuple(selected.ids) != tuple(revealed.ids):
                raise ValueError("Task13 prior selection and oracle labels have different IDs")
            arms = selected.metadata.get("arm_assignments")
            if not isinstance(arms, list) or len(arms) != len(selected):
                raise ValueError("Task13 prior selection lost its arm attributions")
            canonical = [
                self.label_mapping.get(str(raw), str(raw))
                for raw in revealed.labels
            ]
            bandit.update(arms, canonical)
        counts = self._confirmed_object_counts(cycle - 1)
        allocations = bandit.choose(
            batch_size=budget,
            confirmed_counts=counts,
            method=self.config.strategy,
            rng=np.random.default_rng(np.random.SeedSequence(
                [self.config.seed, cycle, 1313]
            )),
        )
        if inference.embeddings is None:
            raise ValueError("Task13 requires object-level final-layer embeddings")
        anchors = self._task11_confirmed_inference(cycle)
        if anchors.embeddings is None:
            raise ValueError("Task13 requires confirmed labeled-object embeddings")
        result = select_mixed_batch(
            object_ids=inference.object_ids,
            predictions=inference.predictions,
            embeddings=inference.embeddings,
            confirmed_counts=counts,
            labeled_probabilities=anchors.predictions.as_probabilities(),
            labeled_embeddings=anchors.embeddings,
            arm_assignments=allocations,
            exploration_rng=np.random.default_rng(np.random.SeedSequence(
                [self.config.seed, cycle, 1314]
            )),
        )
        result.metadata.update({
            "meta_policy": self.config.strategy,
            "meta_posterior_before": bandit.history_summary(),
            "meta_reward_weights": bandit.reward_weights(counts).tolist(),
            "meta_confirmed_object_counts": counts.tolist(),
            "meta_label_source": "previous_confirmed_training_manifests_and_acquired_labels_only",
            "meta_allocation_seed_components": [self.config.seed, cycle, 1313],
            "meta_exploration_seed_components": [self.config.seed, cycle, 1314],
            "meta_settings": {
                "arms": list(settings.arms),
                "discount": settings.discount,
                "prior_concentration": settings.prior_concentration,
                "reward": settings.reward,
            },
        })
        return result

    def _labels_or_resume(
''',
        "simulation-meta-history-and-selection",
    )
    source = replace_once(
        source,
        '        if self.config.strategy == "entropy_balanced_explore":\n'
        '            payload["class_names"] = list(self.class_names)\n',
        '        if self.config.strategy in {"entropy_balanced_explore", "tailor_diversity", "uniform_mixture"}:\n'
        '            payload["class_names"] = list(self.class_names)\n',
        "meta-identity-class-names",
    )
    source = replace_once(
        source,
        '            payload["label_mapping"] = self.label_mapping\n        return payload\n',
        '            payload["label_mapping"] = self.label_mapping\n'
        '        if self.config.strategy in {"tailor_diversity", "uniform_mixture"}:\n'
        '            settings = TailorSettings()\n'
        '            payload["task13_meta_settings"] = {\n'
        '                "arms": list(settings.arms),\n'
        '                "discount": settings.discount,\n'
        '                "prior_concentration": settings.prior_concentration,\n'
        '                "reward": settings.reward,\n'
        '                "batch_policy": asdict(BatchAcquisitionConfig()),\n'
        '            }\n'
        '        return payload\n',
        "meta-identity-settings",
    )
    path.write_text(source, encoding="utf-8")
    return list(paths.values())
