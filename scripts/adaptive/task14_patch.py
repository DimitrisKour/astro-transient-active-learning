"""Strict integration patch for a verified COPY of Task11's frozen source."""
from pathlib import Path
import shutil

POLICIES = ('similar_fl', 'similar_flcg', 'similar_flqmi_rare')


def once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise ValueError(f'Task14 patch mismatch {label}: {count} anchors')
    return text.replace(old, new, 1)


def patch_source(root, overlay):
    root, overlay = Path(root), Path(overlay)
    target = root/'src/al_pipeline/core/advanced_acquisition/similar.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(overlay/'src/al_pipeline/core/advanced_acquisition/similar.py', target)

    p = root/'src/al_pipeline/core/experiments/config.py'
    text = p.read_text()
    text = once(text, 'StrategyName: TypeAlias = Literal[\n',
                'StrategyName: TypeAlias = Literal[\n    "similar_fl", "similar_flcg", "similar_flqmi_rare",\n', 'config-strategy')
    text = once(text, '        if self.strategy not in {\n',
                '        if self.strategy not in {\n            "similar_fl", "similar_flcg", "similar_flqmi_rare",\n', 'config-allowed')
    p.write_text(text)

    p = root/'src/al_pipeline/astronomy/experiments/astrominn.py'
    text = p.read_text()
    text = once(text, '            "badge", "core_set", "fisher_batch",\n',
                '            "badge", "core_set", "fisher_batch",\n            "similar_fl", "similar_flcg", "similar_flqmi_rare",\n', 'suite-allowed')
    p.write_text(text)

    p = root/'scripts/run_astrominn_closed_loop.py'
    text = p.read_text()
    text = once(text, '            "badge", "core_set", "fisher_batch",\n',
                '            "badge", "core_set", "fisher_batch",\n            "similar_fl", "similar_flcg", "similar_flqmi_rare",\n', 'runner-choices')
    text = once(text, '    if set(args.strategies) - {"badge", "core_set", "fisher_batch"}:\n        raise ValueError("Task11 runner permits only the three frozen Task11 acquisition policies")\n',
                '    if set(args.strategies) - {"similar_fl", "similar_flcg", "similar_flqmi_rare"}:\n        raise ValueError("Task14 runner permits only frozen SIMILAR acquisition policies")\n', 'runner-lock')
    p.write_text(text)

    p = root/'src/al_pipeline/astronomy/experiments/simulated.py'
    text = p.read_text()
    text = once(text, 'from al_pipeline.core.advanced_acquisition.fisher_batch import FisherBatchInput, FisherBatchScorer, FisherBatchSelector\n',
                'from al_pipeline.core.advanced_acquisition.fisher_batch import FisherBatchInput, FisherBatchScorer, FisherBatchSelector\n'
                'from al_pipeline.core.advanced_acquisition.similar import SimilarInput, SimilarScorer, SimilarSelector\n',
                'simulation-import')
    text = once(text, '            if self.config.strategy == "badge":\n                selector = BadgeSelector(',
                '            if self.config.strategy in {"similar_fl", "similar_flcg", "similar_flqmi_rare"}:\n'
                '                selector = SimilarSelector(acquisition_data)\n'
                '            elif self.config.strategy == "badge":\n                selector = BadgeSelector(', 'selector')
    text = once(text, '        if self.config.strategy == "badge":\n            return BadgeGradientNormScorer()\n',
                '        if self.config.strategy in {"similar_fl", "similar_flcg", "similar_flqmi_rare"}:\n'
                '            return SimilarScorer()\n'
                '        if self.config.strategy == "badge":\n            return BadgeGradientNormScorer()\n', 'scorer')
    text = once(text, '        if self.config.strategy == "badge":\n            if inference.embeddings is None:\n',
                '''        if self.config.strategy in {"similar_fl", "similar_flcg", "similar_flqmi_rare"}:
            if inference.embeddings is None:
                raise ValueError('SIMILAR requires classifier-input embeddings')
            if self.config.strategy == 'similar_fl':
                return SimilarInput(
                    candidate_ids=tuple(inference.object_ids),
                    candidate_probabilities=inference.predictions.as_probabilities(),
                    candidate_embeddings=inference.embeddings, policy=self.config.strategy)
            anchor = self._task11_confirmed_inference(cycle)
            if anchor.embeddings is None:
                raise ValueError('SIMILAR labeled reference embeddings missing')
            labels = self._task14_confirmed_labels(cycle - 1)
            if set(anchor.object_ids) != set(labels):
                raise ValueError('SIMILAR confirmed-reference IDs disagree with training manifest')
            target_classes = self._task14_fixed_rare_classes() if self.config.strategy == 'similar_flqmi_rare' else ()
            return SimilarInput(
                candidate_ids=tuple(inference.object_ids),
                candidate_probabilities=inference.predictions.as_probabilities(),
                candidate_embeddings=inference.embeddings,
                reference_ids=tuple(anchor.object_ids),
                reference_probabilities=anchor.predictions.as_probabilities(),
                reference_embeddings=anchor.embeddings,
                reference_labels=tuple(labels[oid] for oid in anchor.object_ids),
                rare_classes=target_classes, policy=self.config.strategy)

        if self.config.strategy == "badge":
            if inference.embeddings is None:
''', 'similar-input')
    text = once(text, '    def _confirmed_object_counts(self, through_cycle: int) -> np.ndarray:\n',
                '''    def _task14_confirmed_labels(self, through_cycle: int) -> dict[str, str]:
        """Only known labels in L_t, deduplicated at object level."""
        path = self.labeled_sets.training_manifest(through_cycle)
        labels = {}
        with path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                oid = row[self.dataset.object_id_column]
                raw = row[self.dataset.label_column]
                label = self.label_mapping.get(raw, raw)
                if label not in self.class_names:
                    raise ValueError(f'Unknown confirmed SIMILAR class {raw!r}')
                if oid in labels and labels[oid] != label:
                    raise ValueError('Conflicting labels for SIMILAR reference object '+oid)
                labels[oid] = label
        if not labels:
            raise ValueError('No confirmed labels')
        return labels

    def _task14_fixed_rare_classes(self) -> tuple[str, ...]:
        """Fix target classes from L0, never use validation/hidden labels."""
        labels = self._task14_confirmed_labels(0)
        counts = {name: sum(x == name for x in labels.values()) for name in self.class_names}
        if not all(counts.values()):
            raise ValueError('SIMILAR requires confirmed L0 examples for all classes')
        return tuple(sorted(self.class_names, key=lambda name: (counts[name],
                    self.class_names.index(name)))[:2])

    def _confirmed_object_counts(self, through_cycle: int) -> np.ndarray:
''', 'confirmed-target')
    p.write_text(text)
