"""Balanced exploration seed 357: strict, matched row-order/seed retry experiments.

An ordinary base cycle is committed by the source AL coordinator, but the next
acquisition is blocked until a durable *final* retention decision is recorded.
Each alternate-seed fit is a separate stage. No best-of-k selection: the FIRST
candidate satisfying the original 1 pp macro-F1 rule becomes the incumbent.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections import Counter
import csv
import json
import os
import numpy as np

from al_pipeline.astronomy.experiments.simulated import SimulatedActiveLearningExperiment
from al_pipeline.astronomy.experiments.retention import (
    RetentionExperiment, CLASSES, digest, prediction_file, verified_evaluation,
    read_json, atomic_json,
)
from al_pipeline.core.analysis.retention_gate import paired_retention
from al_pipeline.core.training.base import TrainingRequest
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer

RETRY_SEEDS = (1001, 1002, 1003, 1004)
SCHEMA = 'balanced357_strict_retry_v1'
ROW_SEEDS = (41004, 42004, 43004, 44004)


def strict_accept(incumbent: dict, candidate: dict) -> bool:
    """Strictly greater macro-F1 AND balanced accuracy; ties reject."""
    return (candidate['macro_f1'] > incumbent['macro_f1']
            and candidate['balanced_accuracy'] > incumbent['balanced_accuracy'])


def shuffled_manifest(src: Path, dst: Path, seed: int) -> str:
    """Deterministically reorder rows only; preserve every row and multiplicity."""
    with Path(src).open(newline='', encoding='utf-8') as handle:
        rdr = csv.DictReader(handle)
        fields = rdr.fieldnames
        if not fields:
            raise ValueError('Empty source training manifest')
        rows = list(rdr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    if np.array_equal(perm, np.arange(len(rows))):
        raise ValueError('Identity permutation')
    shuffled = [rows[int(i)] for i in perm]
    def multiset(items):
        return Counter(tuple(row[field] for field in fields) for row in items)
    if multiset(rows) != multiset(shuffled):
        raise ValueError('Row permutation modified training sample multiset')
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(dst.name + f'.tmp.{os.getpid()}')
    try:
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(shuffled)
        if dst.exists():
            if digest(dst) != digest(temporary):
                raise ValueError('Existing row permutation disagrees with frozen seed')
        else:
            temporary.replace(dst)
        return digest(dst)
    finally:
        temporary.unlink(missing_ok=True)


class Balanced357RetryExperiment(RetentionExperiment):
    """Reference Task4 prefix + strict two-metric retention with 4 matched retries.

    After four failed retries, the incumbent stays active but all acquired
    labels remain in L_t. The next cycle's inference and training start from
    that incumbent through RetentionExperiment._incumbent.
    """

    def __init__(self, *, retry_mode='row_order', **kwargs):
        super().__init__(**kwargs)
        if retry_mode not in ('row_order', 'seed'):
            raise ValueError('Unknown retry mode')
        self.retry_mode = retry_mode

    @classmethod
    def from_experiment(cls, experiment, *, reference_persistent, reference_data, retry_mode):
        from al_pipeline.core.analysis.retention_gate import RetentionRule
        obj = super().from_experiment(
            experiment, reference_persistent=reference_persistent,
            reference_data=reference_data, rule=RetentionRule(macro_tolerance=0.0))
        if retry_mode not in ('row_order', 'seed'):
            raise ValueError('Unknown retry mode')
        obj.retry_mode = retry_mode
        if obj.config.strategy != 'entropy_balanced_explore' or obj.config.seed != 357:
            raise ValueError('This experiment is restricted to balanced exploration seed 357')
        return obj

    def _identity_payload(self):
        value = super()._identity_payload()
        value['retention']['schema'] = SCHEMA
        value['retention']['retry_mode'] = self.retry_mode
        value['retention']['alternative_seeds'] = list(RETRY_SEEDS) if self.retry_mode == 'seed' else []
        value['retention']['row_order_seeds'] = list(ROW_SEEDS) if self.retry_mode == 'row_order' else []
        value['retention']['active_rule'] = 'strict_both_macro_f1_and_balanced_accuracy'
        value['retention']['acceptance'] = 'first_strictly_passing_retry_else_retain'
        return value

    def _comparison(self, cycle: int, checkpoint: Path, eval_dir: Path):
        # RetentionExperiment has no _comparison. Align object IDs and targets
        # via the shared paired audit, then impose the strict two-metric gate.
        parent, parent_dir, _ = self._incumbent(cycle)
        verified_evaluation(parent_dir, parent)
        verified_evaluation(eval_dir, checkpoint)
        result = paired_retention(
            prediction_file(parent_dir / 'validation_predictions.npz'),
            prediction_file(eval_dir / 'validation_predictions.npz'),
            self.rule, CLASSES)
        result['accepted'] = strict_accept(
            result['incumbent_metrics'], result['candidate_metrics'])
        result['decision_rule'] = 'strict_both_macro_f1_and_balanced_accuracy'
        return result

    def _base_path(self, cycle: int) -> Path:
        return self.run_dir / f'cycle_{cycle:03d}' / 'base_attempt.json'

    def _retry_path(self, cycle: int, number: int) -> Path:
        return self.run_dir / f'cycle_{cycle:03d}' / 'retry_attempts' / f'retry_{number}.json'

    def _retry_dir(self, cycle: int, number: int) -> Path:
        tag = (f'row_{ROW_SEEDS[number-1]}' if self.retry_mode == 'row_order'
               else f'seed_{RETRY_SEEDS[number-1]}')
        return self.artifact_dir / f'cycle_{cycle:03d}' / 'retries' / tag

    def _assert_prior_final(self, cycle: int) -> None:
        state = self.initialize()
        if cycle < 1 or cycle > self.config.max_cycles:
            raise ValueError('Cycle outside configured acquisition budget')
        if state.last_completed_cycle not in (cycle-1, cycle):
            raise RuntimeError('Attempted out-of-order cycle or unfinished predecessor')
        if state.active_cycle not in (None,cycle):
            raise RuntimeError('Different AL cycle is active')
        for c in range(1,cycle):
            if not self._decision_path(c).is_file():
                raise RuntimeError(f'Prior cycle {c} not finalized; refusing acquisition')

    @staticmethod
    def _save_checked(path: Path, record: dict) -> None:
        if path.exists():
            if read_json(path) != record:
                raise ValueError(f'Existing stage record differs: {path}')
        else:
            atomic_json(path,record)

    def _source_base(self, cycle: int) -> tuple[Path,Path,Path]:
        src = read_json(self.run_dir/f'cycle_{cycle:03d}'/'cycle_summary.json')
        checkpoint = Path(src['trained_checkpoint'])
        eval_dir = Path(src['evaluation_summary']).parent
        manifest = Path(src['cumulative_manifest'])
        if not manifest.is_file():raise FileNotFoundError(manifest)
        verified_evaluation(eval_dir,checkpoint)
        return checkpoint,eval_dir,manifest

    def _base_record(self, cycle: int) -> dict:
        checkpoint,eval_dir,manifest=self._source_base(cycle)
        comparison=self._comparison(cycle,checkpoint,eval_dir)
        parent,_,_=self._incumbent(cycle)
        record={'schema':SCHEMA,'cycle':cycle,'kind':'base',
                'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
                'evaluation_dir':str(eval_dir),
                'evaluation_sha256':digest(eval_dir/'evaluation_summary.json'),
                'prediction_sha256':digest(eval_dir/'validation_predictions.npz'),
                'training_manifest':str(manifest),'training_manifest_sha256':digest(manifest),
                'initial_checkpoint':str(parent),'initial_checkpoint_sha256':digest(parent),
                'reference_reused':checkpoint.is_relative_to(self.reference_heavy()),
                'comparison':comparison}
        self._save_checked(self._base_path(cycle),record)
        return record

    def _finalize(self,cycle:int, selected:dict, accepted:bool, retry_number:int|None)->dict:
        base=read_json(self._base_path(cycle))
        parent,parent_eval,source_cycle=self._incumbent(cycle)
        check=Path(selected['checkpoint'])
        evdir=Path(selected['evaluation_dir'])
        verified_evaluation(evdir,check)
        selected_comparison=self._comparison(cycle,check,evdir)
        if selected_comparison != selected['comparison']:
            raise ValueError('Selected comparison differs from persisted attempt')
        if accepted != selected_comparison['accepted']:
            raise ValueError('Final acceptance contradicts strict gate')
        chosen=check if accepted else parent
        chosen_eval=evdir if accepted else parent_eval
        summary=read_json(self.run_dir/f'cycle_{cycle:03d}'/'cycle_summary.json')
        decision={'schema':SCHEMA,'cycle':cycle,'seed':self.config.seed,
                  'policy':self.config.strategy,'active_rule':'strict_both_macro_f1_and_balanced_accuracy',
                  'retry_mode':self.retry_mode,
                  'accepted':bool(accepted),'accepted_retry_number':retry_number if accepted else None,
                  'accepted_retry_seed':(RETRY_SEEDS[retry_number-1] if self.retry_mode == 'seed' and accepted and retry_number else None),
                  'attempts_consumed':0 if accepted and retry_number is None else (
                       retry_number if accepted else len(RETRY_SEEDS)),
                  'reference_reused':base['reference_reused'],
                  'base_candidate_checkpoint':base['checkpoint'],
                  'base_comparison':base['comparison'],
                  'candidate_checkpoint':str(check),'candidate_sha256':digest(check),
                  'candidate_evaluation_dir':str(evdir),
                  'candidate_validation_npz_sha256':digest(evdir/'validation_predictions.npz'),
                  'parent_checkpoint':str(parent),'parent_sha256':digest(parent),
                  'parent_source_cycle':source_cycle,
                  'incumbent_checkpoint':str(chosen),'incumbent_sha256':digest(chosen),
                  'incumbent_evaluation_dir':str(chosen_eval),
                  'incumbent_source_cycle':cycle if accepted else source_cycle,
                  'training_manifest':summary['cumulative_manifest'],
                  'training_manifest_sha256':digest(Path(summary['cumulative_manifest'])),
                  'acquired_this_cycle':len(summary['selected_ids']),
                  'comparison':selected_comparison}
        self._save_checked(self._decision_path(cycle), decision)
        return decision

    def base_stage(self,cycle:int)->dict:
        self._assert_prior_final(cycle)
        state=self.initialize()
        if state.last_completed_cycle==cycle:
            if state.active_cycle is not None:raise RuntimeError('Completed cycle with active state')
        else:
            self._reference_cycle=cycle if self._can_reference_replay(cycle) else None
            try:
                SimulatedActiveLearningExperiment.run_next_cycle(self)
            finally:
                self._reference_cycle=None
        record=self._base_record(cycle)
        if record['comparison']['accepted']:
            self._finalize(cycle,record,True,None)
        elif self._decision_path(cycle).is_file():
            # Previously finalized by retry worker. Never reassess differently.
            self.check_final(cycle)
        return record

    def check_final(self,cycle:int)->dict:
        row=read_json(self._decision_path(cycle))
        if row['schema']!=SCHEMA or row['cycle']!=cycle or row['seed']!=self.config.seed or row['policy']!=self.config.strategy:
            raise ValueError('Final decision identity mismatch')
        chosen=Path(row['incumbent_checkpoint'])
        if digest(chosen)!=row['incumbent_sha256']:
            raise ValueError('Final incumbent checkpoint content changed')
        verified_evaluation(Path(row['incumbent_evaluation_dir']),chosen)
        if digest(Path(row['training_manifest']))!=row['training_manifest_sha256']:
            raise ValueError('Cumulative L_t changed since final decision')
        return row

    def _verified_retry(self, cycle:int,number:int)->dict:
        path=self._retry_path(cycle,number)
        row=read_json(path)
        parent,_,_=self._incumbent(cycle)
        base=read_json(self._base_path(cycle))
        checkpoint=Path(row['checkpoint'])
        eval_dir=Path(row['evaluation_dir'])
        expect_seed = self.config.seed if self.retry_mode == 'row_order' else RETRY_SEEDS[number-1]
        expect_order = ROW_SEEDS[number-1] + cycle * 100 if self.retry_mode == 'row_order' else None
        if (row['cycle'] != cycle or row['retry_number'] != number
            or row['retry_seed'] != expect_seed or row['schema'] != SCHEMA
            or row.get('retry_mode') != self.retry_mode
            or row.get('permutation_seed') != expect_order
            or row.get('loader_seed') != expect_seed):
            raise ValueError('Retry stage identity mismatch')
        manifest = Path(row['retry_manifest'])
        if digest(manifest) != row['retry_manifest_sha256']:
            raise ValueError('Retry manifest content changed')
        if self.retry_mode == 'row_order':
            checked = shuffled_manifest(Path(base['training_manifest']), manifest, expect_order)
            if checked != row['retry_manifest_sha256']:
                raise ValueError('Row-order permutation changed')
        elif digest(manifest) != digest(Path(base['training_manifest'])):
            raise ValueError('Seed retry unexpectedly changed training rows')
        if (digest(checkpoint)!=row['checkpoint_sha256'] or
            digest(parent)!=row['initial_checkpoint_sha256'] or
            digest(Path(base['training_manifest']))!=row['training_manifest_sha256']):
            raise ValueError('Retry provenance changed')
        verified_evaluation(eval_dir,checkpoint)
        if (digest(eval_dir/'evaluation_summary.json')!=row['evaluation_sha256'] or
            digest(Path(row['training_summary']))!=row['training_summary_sha256']):
            raise ValueError('Retry training/evaluation summary changed')
        if self._comparison(cycle,checkpoint,eval_dir)!=row['comparison']:
            raise ValueError('Retry validation comparison changed')
        return row

    def retry_stage(self,cycle:int,number:int)->dict:
        if number not in range(1,len(RETRY_SEEDS)+1):
            raise ValueError('Retry index outside frozen budget')
        self._assert_prior_final(cycle)
        state=self.initialize()
        if state.last_completed_cycle!=cycle or state.active_cycle is not None:
            raise RuntimeError('Base cycle not completed')
        base=read_json(self._base_path(cycle))
        if base['comparison']['accepted']:
            self.check_final(cycle)
            return {'skipped':'base_accepted','cycle':cycle,'retry_number':number}
        # Validate order even when an earlier retry succeeded.
        for prior in range(1,number):
            if not self._retry_path(cycle,prior).is_file():
                final=self._decision_path(cycle)
                if final.is_file() and self.check_final(cycle)['accepted']:
                    return {'skipped':'earlier_retry_accepted','cycle':cycle,'retry_number':number}
                raise RuntimeError('Earlier retry missing; refusing out-of-order attempt')
            prev=self._verified_retry(cycle,prior)
            if prev['comparison']['accepted']:
                self.check_final(cycle)
                return {'skipped':'earlier_retry_accepted','cycle':cycle,'retry_number':number}
        if self._retry_path(cycle,number).is_file():
            row=self._verified_retry(cycle,number)
        else:
            parent,_,_=self._incumbent(cycle)
            output = self._retry_dir(cycle, number)
            if self.retry_mode == 'row_order':
                seed = self.config.seed
                order_seed = ROW_SEEDS[number-1] + cycle * 100
                retry_manifest = output / 'train_manifest_permuted.csv'
                shuffled_manifest(Path(base['training_manifest']), retry_manifest, order_seed)
            else:
                seed = RETRY_SEEDS[number-1]
                order_seed = None
                retry_manifest = Path(base['training_manifest'])
            request = TrainingRequest(
                train_manifest=retry_manifest,
                validation_manifest=Path(self.dataset.validation_manifest),
                output_dir=output/'training',
                run_id=f'{self.config.run_id}_{self.retry_mode}_retry_{cycle}_{number}_{seed}',
                cycle=cycle, strategy=self.config.strategy, seed=seed,
                initial_checkpoint=parent)
            policy=replace(self.trainer.policy,loader_seed=seed)
            if (policy.class_weight_exponent!=.85 or policy.new_object_weighting!='per_sample'
                or policy.class_weight_source!='training_manifest' or policy.overwrite):
                raise ValueError('Retry scientific configuration differs')
            trainer=AstroMiNNViTTrainer(dataset=self.dataset,policy=policy,
                device=self.trainer.device,initialization=self.trainer.initialization,
                legacy_code_dir=self.trainer.legacy_code_dir,
                artifact_dir=self.trainer.artifact_dir,
                model_factory=self.trainer.model_factory)
            done=trainer.completed_result(request)
            if done is None:
                if request.output_dir.exists() and any(request.output_dir.iterdir()):
                    raise RuntimeError(f'Partial retry training output: {request.output_dir}')
                done=trainer.train(request)
            evaluation=self.evaluator.completed_result(
                checkpoint_path=done.checkpoint_path, output_dir=output/'evaluation')
            if evaluation is None:
                evaluation=self.evaluator.evaluate(
                    checkpoint_path=done.checkpoint_path,output_dir=output/'evaluation')
            eval_dir=evaluation.summary_path.parent
            compare=self._comparison(cycle,done.checkpoint_path,eval_dir)
            row={'schema':SCHEMA,'cycle':cycle,'retry_number':number,'retry_seed':seed,
                  'retry_mode':self.retry_mode,'loader_seed':seed,
                  'permutation_seed':order_seed,
                  'retry_manifest':str(retry_manifest),
                  'retry_manifest_sha256':digest(retry_manifest),
                 'checkpoint':str(done.checkpoint_path),
                 'checkpoint_sha256':digest(done.checkpoint_path),
                 'evaluation_dir':str(eval_dir),
                 'evaluation_sha256':digest(eval_dir/'evaluation_summary.json'),
                 'training_summary':str(done.summary_path),
                 'training_summary_sha256':digest(done.summary_path),
                 'initial_checkpoint_sha256':digest(parent),
                 'training_manifest_sha256':digest(Path(base['training_manifest'])),
                 'elapsed_training_seconds':done.elapsed_seconds,
                 'comparison':compare}
            self._save_checked(self._retry_path(cycle,number),row)
        if row['comparison']['accepted']:
            self._finalize(cycle,row,True,number)
        elif number==len(RETRY_SEEDS):
            self._finalize(cycle,row,False,None)
        return row
