"""Task8C.1 retention-only extension of frozen simulated AstroMiNN AL.

No production trainer/evaluator edits. Historical Task4 cycles are replayed only
while all preceding candidates have been accepted, with hash/provenance checks.
After first rejection, inference and training use the last ACCEPTED checkpoint,
while cumulative L_t continues expanding independently of model acceptance.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from al_pipeline.astronomy.experiments.simulated import SimulatedActiveLearningExperiment
from al_pipeline.core.training.base import TrainingResult
from al_pipeline.core.evaluation.base import ModelEvaluationResult, SplitEvaluation
from al_pipeline.core.analysis.retention_gate import ObjectPredictions, RetentionRule, paired_retention

CLASSES = ('AGN', 'TDE', 'SN I', 'SN II', 'CV')
REFERENCE_SUITE = 'astrominn_closed_loop_warm_v1'
SCHEMA = 'task8c1_retention_v1'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    if tmp.exists():
        raise RuntimeError(f'Unresolved incomplete write: {tmp}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def prediction_file(path: Path) -> ObjectPredictions:
    with np.load(path, allow_pickle=False) as f:
        if not {'object_ids','object_targets','object_values'}.issubset(f.files):
            raise ValueError(f'Not an evaluation prediction NPZ: {path}')
        return ObjectPredictions(tuple(np.asarray(f['object_ids'], dtype=str)),
                                 np.asarray(f['object_targets'], dtype=int),
                                 np.asarray(f['object_values'], dtype=float))


def verified_evaluation(eval_dir: Path, checkpoint: Path) -> tuple[dict, Path]:
    summary = read_json(eval_dir / 'evaluation_summary.json')
    pred = eval_dir / 'validation_predictions.npz'
    if not pred.is_file():
        raise FileNotFoundError(pred)
    if (Path(summary['checkpoint_path']) != checkpoint or
        summary['checkpoint_sha256'] != digest(checkpoint) or
        tuple(summary['model_class_names']) != CLASSES or
        not any(row['split'] == 'validation' for row in summary['splits'])):
        raise ValueError(f'Checkpoint/evaluation provenance mismatch: {eval_dir}')
    return summary, pred


class RetentionExperiment(SimulatedActiveLearningExperiment):
    """Resumable experiment with an independent immutable incumbent ledger."""

    def __init__(self, *, reference_persistent: Path, reference_data: Path,
                 rule: RetentionRule = RetentionRule(), **kwargs):
        super().__init__(**kwargs)
        if self.config.training_initialization != 'warm_start':
            raise ValueError('Task8C.1 requires warm_start')
        if self.config.strategy not in ('entropy','entropy_balanced_explore'):
            raise ValueError('Task8C.1 supports only the two approved policies')
        if tuple(self.class_names) != CLASSES:
            raise ValueError('Unexpected AstroMiNN class mapping')
        self.reference_persistent = Path(reference_persistent)
        self.reference_data = Path(reference_data)
        self.rule = rule
        self._reference_cycle: int | None = None

    @classmethod
    def from_experiment(cls, experiment: SimulatedActiveLearningExperiment,
                        *, reference_persistent: Path, reference_data: Path,
                        rule: RetentionRule = RetentionRule()):
        return cls(config=experiment.config, dataset=experiment.dataset,
                   pool_path=experiment.pool_path, hidden_labels=experiment.hidden_labels,
                   run_dir=experiment.run_dir, artifact_dir=experiment.artifact_dir,
                   baseline_checkpoint=experiment.baseline_checkpoint,
                   predictor=experiment.predictor, trainer=experiment.trainer,
                   evaluator=experiment.evaluator, holdout_ids=experiment.holdout_ids,
                   class_names=experiment.class_names, label_mapping=experiment.label_mapping,
                   reference_persistent=reference_persistent,
                   reference_data=reference_data, rule=rule)

    def _identity_payload(self):
        value = super()._identity_payload()
        value['retention'] = {'schema': SCHEMA, 'active_rule': 'macro_f1_1pp',
                              'macro_tolerance': self.rule.macro_tolerance,
                              'shadow_ba_tolerance': self.rule.balanced_accuracy_shadow_tolerance,
                              'shadow_class_limits': list(self.rule.shadow_net_correct_loss_limits),
                              'reference_suite': REFERENCE_SUITE,
                              'reference_persistent': str(self.reference_persistent),
                              'reference_data': str(self.reference_data)}
        return value

    def reference_state(self) -> Path:
        return self.reference_persistent / 'runs' / REFERENCE_SUITE / self.config.strategy / f'seed_{self.config.seed}'

    def reference_heavy(self) -> Path:
        return self.reference_data / 'runs' / REFERENCE_SUITE / self.config.strategy / f'seed_{self.config.seed}'

    def _source_summary(self, cycle: int) -> dict:
        return read_json(self.reference_state() / f'cycle_{cycle:03d}' / 'cycle_summary.json')

    def _decision_path(self, cycle: int) -> Path:
        return self.run_dir / f'cycle_{cycle:03d}' / 'retention_decision.json'

    def _incumbent(self, cycle: int) -> tuple[Path, Path, int]:
        """Checkpoint, validation evaluation directory, source accepted cycle."""
        if cycle == 1:
            return (self.baseline_checkpoint,
                    self.reference_heavy() / 'cycle_000' / 'evaluation', 0)
        prev = self._decision_path(cycle - 1)
        if not prev.is_file():
            raise RuntimeError(f'Missing predecessor acceptance decision: {prev}')
        row = read_json(prev)
        path = Path(row['incumbent_checkpoint'])
        if digest(path) != row['incumbent_sha256']:
            raise ValueError(f'Incumbent checkpoint changed: {path}')
        return path, Path(row['incumbent_evaluation_dir']), int(row['incumbent_source_cycle'])

    def _can_reference_replay(self, cycle: int) -> bool:
        # No replay after *any* rejected checkpoint, even if the retained source
        # checkpoint happens to have the same path as some later reference.
        for c in range(1, cycle):
            row = read_json(self._decision_path(c))
            if not row['accepted']:
                return False
        checkpoint, _, _ = self._incumbent(cycle)
        expected = self.baseline_checkpoint if cycle == 1 else (
            self.reference_heavy() / f'cycle_{cycle-1:03d}' / 'training' / 'best_vit_tower.pth')
        return checkpoint.resolve() == expected.resolve()

    def _inference_checkpoint(self, cycle: int) -> Path:
        return self._incumbent(cycle)[0]

    def _training_checkpoint(self, cycle: int) -> Path:
        return self._incumbent(cycle)[0]

    def evaluate_baseline(self):
        self.initialize()
        summary, _ = verified_evaluation(self.reference_heavy() / 'cycle_000' / 'evaluation',
                                         self.baseline_checkpoint)
        validation = next(row for row in summary['splits'] if row['split'] == 'validation')
        result = ModelEvaluationResult(checkpoint_path=self.baseline_checkpoint,
            summary_path=self.reference_heavy()/'cycle_000'/'evaluation'/'evaluation_summary.json',
            splits=(SplitEvaluation(**validation),))
        self._record_evaluation(cycle=0, n_acquired=0, result=result)
        return result

    def _copy_reference(self, src: Path, dst: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if digest(dst) != digest(src):
                raise ValueError(f'Previously persisted reference file differs: {dst}')
        else:
            shutil.copy2(src, dst)
            if digest(dst) != digest(src):
                raise ValueError(f'Reference transfer corrupted: {src}')

    def _infer_or_resume(self, *, object_ids, checkpoint_path, path):
        if self._reference_cycle is not None:
            c = self._reference_cycle
            summary = self._source_summary(c)
            if Path(summary['inference_checkpoint']) != checkpoint_path:
                raise ValueError(f'Task4 inference checkpoint ancestry mismatch at cycle {c}')
            self._copy_reference(self.reference_heavy()/f'cycle_{c:03d}'/'inference.npz', path)
        return super()._infer_or_resume(object_ids=object_ids, checkpoint_path=checkpoint_path, path=path)

    def _select_or_resume(self, *, inference, budget, cycle, path):
        if self._reference_cycle is not None:
            source = self.reference_state()/f'cycle_{cycle:03d}'
            self._copy_reference(source/'selection.json', path)
            if (source/'candidate_scores.npz').is_file():
                self._copy_reference(source/'candidate_scores.npz', path.with_name('candidate_scores.npz'))
        result = super()._select_or_resume(inference=inference, budget=budget, cycle=cycle, path=path)
        if self._reference_cycle is not None and list(result.ids) != self._source_summary(cycle)['selected_ids']:
            raise ValueError('Reference acquisition selection changed')
        return result

    def _labels_or_resume(self, *, selection, path):
        if self._reference_cycle is not None:
            self._copy_reference(self.reference_state()/f'cycle_{self._reference_cycle:03d}'/'revealed_labels.json', path)
        return super()._labels_or_resume(selection=selection, path=path)

    def _train_or_resume(self, request):
        if self._reference_cycle is None:
            # No implicit restart of a partially written training directory.
            if (request.output_dir / 'training_summary.json').exists():
                return super()._train_or_resume(request)
            if request.output_dir.is_dir() and any(request.output_dir.iterdir()):
                raise RuntimeError(f'Partial training requires manual recovery: {request.output_dir}')
            return super()._train_or_resume(request)
        c = self._reference_cycle
        root = self.reference_heavy()/f'cycle_{c:03d}'/'training'
        summary_path = root/'training_summary.json'
        record = read_json(summary_path)
        original = self._source_summary(c)
        if (int(record['cycle']) != c or int(record['seed']) != self.config.seed or
            record['strategy'] != self.config.strategy or
            Path(record['initial_checkpoint']) != request.initial_checkpoint or
            Path(record['checkpoint_path']) != Path(original['trained_checkpoint']) or
            digest(request.train_manifest) != record['train_manifest_sha256'] or
            digest(request.validation_manifest) != record['validation_manifest_sha256']):
            raise ValueError(f'Historical checkpoint training input mismatch: cycle {c}')
        policy = json.loads(json.dumps(asdict(self.trainer.policy)))
        if record['policy'] != policy or record['initialization'] != 'pretrained':
            raise ValueError('Frozen Task4 training policy differs from 8C.1')
        result = TrainingResult(checkpoint_path=Path(record['checkpoint_path']),
                                history_path=Path(record['history_path']), summary_path=summary_path,
                                best_epoch=int(record['best_epoch']),
                                epochs_completed=int(record['epochs_completed']),
                                best_validation_loss=float(record['best_validation_loss']),
                                elapsed_seconds=float(record['elapsed_seconds']),
                                peak_cuda_memory_bytes=record.get('peak_cuda_memory_bytes'))
        if not result.checkpoint_path.is_file() or digest(result.checkpoint_path) != read_json(
                self.reference_heavy()/f'cycle_{c:03d}'/'evaluation'/'evaluation_summary.json')['checkpoint_sha256']:
            raise ValueError('Task4 replay checkpoint hash mismatch')
        return result

    def _evaluate_or_resume(self, *, checkpoint_path, output_dir):
        if self._reference_cycle is not None:
            eval_dir = self.reference_heavy()/f'cycle_{self._reference_cycle:03d}'/'evaluation'
            summary, _ = verified_evaluation(eval_dir, checkpoint_path)
            val = next(row for row in summary['splits'] if row['split'] == 'validation')
            return ModelEvaluationResult(checkpoint_path=checkpoint_path,
                summary_path=eval_dir/'evaluation_summary.json', splits=(SplitEvaluation(**val),))
        return super()._evaluate_or_resume(checkpoint_path=checkpoint_path, output_dir=output_dir)

    def _finish_decision(self, cycle: int) -> dict:
        path = self._decision_path(cycle)
        summary = read_json(self.run_dir/f'cycle_{cycle:03d}'/'cycle_summary.json')
        parent, parent_eval, source_c = self._incumbent(cycle)
        candidate = Path(summary['trained_checkpoint'])
        eval_dir = Path(summary['evaluation_summary']).parent
        verified_evaluation(eval_dir, candidate)
        verified_evaluation(parent_eval, parent)
        p_candidate = prediction_file(eval_dir/'validation_predictions.npz')
        p_parent = prediction_file(parent_eval/'validation_predictions.npz')
        result = paired_retention(p_parent, p_candidate, self.rule, CLASSES)
        accepted = result['accepted']
        new_checkpoint = candidate if accepted else parent
        new_eval = eval_dir if accepted else parent_eval
        value = {'schema': SCHEMA, 'cycle': cycle, 'seed': self.config.seed,
                 'policy': self.config.strategy, 'active_rule': 'macro_f1_1pp',
                 'accepted': accepted, 'reference_reused': (candidate.is_relative_to(self.reference_heavy())),
                 'candidate_checkpoint': str(candidate), 'candidate_sha256': digest(candidate),
                 'candidate_evaluation_dir': str(eval_dir),
                 'candidate_validation_npz_sha256': digest(eval_dir/'validation_predictions.npz'),
                 'parent_checkpoint': str(parent), 'parent_sha256': digest(parent),
                 'parent_source_cycle': source_c,
                 'incumbent_checkpoint': str(new_checkpoint),
                 'incumbent_evaluation_dir': str(new_eval),
                 'incumbent_sha256': digest(new_checkpoint),
                 'incumbent_source_cycle': cycle if accepted else source_c,
                 'training_manifest': summary['cumulative_manifest'],
                 'training_manifest_sha256': digest(Path(summary['cumulative_manifest'])),
                 'acquired_this_cycle': len(summary['selected_ids']),
                 'comparison': result}
        if path.exists():
            if read_json(path) != value:
                raise ValueError(f'Existing retention decision does not match recalculation: {path}')
        else:
            atomic_json(path, value)
        return value

    def recover_completed_decisions(self):
        state = self.initialize()
        for c in range(1, state.last_completed_cycle + 1):
            # Decisions may be missing if SLURM killed a worker after writing
            # the ordinary cycle summary but before writing the decision.
            self._finish_decision(c)
        return state

    def run_next_cycle(self):
        state = self.recover_completed_decisions()
        current = state.active_cycle if state.active_cycle is not None else state.last_completed_cycle + 1
        if current > self.config.max_cycles:
            raise RuntimeError('All cycles complete')
        self._reference_cycle = current if self._can_reference_replay(current) else None
        try:
            completed = super().run_next_cycle()
            decision = self._finish_decision(current)
            if completed.cycle != current:
                raise RuntimeError('Completed cycle differs from planned cycle')
            return completed
        finally:
            self._reference_cycle = None

    def run(self):
        self.evaluate_baseline()
        return super().run()
