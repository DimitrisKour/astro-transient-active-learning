"""Task 8C.2: isolated, resumable bounded-retry retention closed loop.

An ordinary base cycle is committed by the source AL coordinator, but the next
acquisition is blocked until a durable *final* retention decision is recorded.
Each alternate-seed fit is a separate stage. No best-of-k selection: the FIRST
candidate satisfying the original 1 pp macro-F1 rule becomes the incumbent.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

from al_pipeline.astronomy.experiments.simulated import SimulatedActiveLearningExperiment
from al_pipeline.astronomy.experiments.retention import (
    RetentionExperiment, CLASSES, digest, prediction_file, verified_evaluation,
    read_json, atomic_json,
)
from al_pipeline.core.analysis.retention_gate import paired_retention
from al_pipeline.core.training.base import TrainingRequest
from al_pipeline.astronomy.astrominn.training import AstroMiNNViTTrainer

RETRY_SEEDS = (1001, 1002, 1003, 1004)
SCHEMA = 'task8c2_bounded_retry_v1'


class RetryRetentionExperiment(RetentionExperiment):
    """Identical acquisition and base training to 8C1; only rejections differ."""

    def _identity_payload(self):
        value = super()._identity_payload()
        value['retention']['schema'] = SCHEMA
        value['retention']['maximum_alternative_training_seeds'] = list(RETRY_SEEDS)
        value['retention']['acceptance'] = 'first_passing_alternative_else_retain'
        return value

    def _base_path(self, cycle: int) -> Path:
        return self.run_dir / f'cycle_{cycle:03d}' / 'base_attempt.json'

    def _retry_path(self, cycle: int, number: int) -> Path:
        return self.run_dir / f'cycle_{cycle:03d}' / 'retry_attempts' / f'retry_{number}.json'

    def _retry_dir(self, cycle: int, number: int) -> Path:
        return self.artifact_dir / f'cycle_{cycle:03d}' / 'retries' / f'seed_{RETRY_SEEDS[number-1]}'

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

    def _comparison(self, cycle: int, checkpoint: Path, eval_dir: Path):
        parent, parent_dir, _ = self._incumbent(cycle)
        verified_evaluation(parent_dir,parent)
        verified_evaluation(eval_dir,checkpoint)
        comparison = paired_retention(
            prediction_file(parent_dir/'validation_predictions.npz'),
            prediction_file(eval_dir/'validation_predictions.npz'),
            self.rule, CLASSES,
        )
        return comparison

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
            raise ValueError('Final acceptance contradicts 1 pp gate')
        chosen=check if accepted else parent
        chosen_eval=evdir if accepted else parent_eval
        summary=read_json(self.run_dir/f'cycle_{cycle:03d}'/'cycle_summary.json')
        decision={'schema':SCHEMA,'cycle':cycle,'seed':self.config.seed,
                  'policy':self.config.strategy,'active_rule':'macro_f1_1pp',
                  'accepted':bool(accepted),'accepted_retry_number':retry_number if accepted else None,
                  'accepted_retry_seed':RETRY_SEEDS[retry_number-1] if accepted and retry_number else None,
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
        if row['cycle']!=cycle or row['retry_number']!=number or row['retry_seed']!=RETRY_SEEDS[number-1] or row['schema']!=SCHEMA:
            raise ValueError('Retry stage identity mismatch')
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
            seed=RETRY_SEEDS[number-1]
            output=self._retry_dir(cycle,number)
            request=TrainingRequest(
                train_manifest=Path(base['training_manifest']),
                validation_manifest=Path(self.dataset.validation_manifest),
                output_dir=output/'training',
                run_id=f'{self.config.run_id}_retry_{cycle}_{seed}',
                cycle=cycle,strategy=self.config.strategy,seed=seed,
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
