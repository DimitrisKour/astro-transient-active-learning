"""Read-only validation of Task 4 artifacts used by first-rejection seed retries."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

from al_pipeline.core.analysis.seed_recovery import RecoveryCase, sha256, SOURCE_SUITE

BOOTSTRAP = 'astrominn_historical_replay_v1'
CLASSES = ('AGN', 'TDE', 'SN I', 'SN II', 'CV')


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


@dataclass(frozen=True)
class VerifiedInputs:
    case: RecoveryCase
    training_manifest: Path
    validation_manifest: Path
    l0_manifest: Path
    initial_checkpoint: Path
    candidate_checkpoint: Path
    source_train_summary: Path
    source_evaluation: Path
    incumbent_evaluation: Path
    model_artifact_dir: Path
    manifest_sha256: str
    validation_sha256: str
    initial_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    source_training_summary_sha256: str
    source_candidate_evaluation_sha256: str
    source_incumbent_evaluation_sha256: str
    model_policy: dict

    def to_dict(self) -> dict:
        fields = vars(self) if hasattr(self, '__dict__') else {
            name: getattr(self, name) for name in self.__dataclass_fields__}
        result = {k: str(v) if isinstance(v, Path) else v for k, v in fields.items()}
        result['case'] = self.case.as_dict()
        return result


def verify_source_case(case: RecoveryCase, persistent: Path, data: Path,
                       registry: Path, *, check_full_hashes: bool = True) -> VerifiedInputs:
    """Refuse stale, mismatched, or incomplete saved Task4 checkpoint lineage."""
    # Imports local to allow pure CPU-only tests and planning on login nodes.
    from al_pipeline.astronomy.data.registry import DatasetRegistry
    resolved = DatasetRegistry.from_toml(registry).resolve_profile('astrominn_ztf')
    original_dataset = resolved.dataset
    validation = Path(original_dataset.validation_manifest)
    source_run = persistent / 'runs' / SOURCE_SUITE / case.policy / f'seed_{case.seed}'
    source_data = data / 'runs' / SOURCE_SUITE / case.policy / f'seed_{case.seed}'
    source_summary_path = source_run / f'cycle_{case.rejected_cycle:03d}' / 'cycle_summary.json'
    source_summary = load_json(source_summary_path)
    prior_cycle_summary = load_json(source_run / f'cycle_{case.incumbent_cycle:03d}' / 'cycle_summary.json')
    previous_eval_path = source_data / f'cycle_{case.incumbent_cycle:03d}' / 'evaluation' / 'evaluation_summary.json'
    candidate_eval_path = source_data / f'cycle_{case.rejected_cycle:03d}' / 'evaluation' / 'evaluation_summary.json'
    incumbent_eval, candidate_eval = load_json(previous_eval_path), load_json(candidate_eval_path)
    current_training = source_data / f'cycle_{case.rejected_cycle:03d}' / 'training' / 'training_summary.json'
    train_record = load_json(current_training)
    manifest = Path(source_summary['cumulative_manifest'])
    known_manifest = source_run / 'manifests' / f'train_cycle_{case.rejected_cycle:03d}.csv'
    l0 = persistent / 'runs' / SOURCE_SUITE / 'shared' / 'historical_replay_split' / 'initial_labeled_manifest.csv'
    source_checkpoint = Path(candidate_eval['checkpoint_path'])
    initial_checkpoint = Path(incumbent_eval['checkpoint_path'])
    model_checkpoint = resolved.profile.options.get('checkpoint')
    if not model_checkpoint:
        raise ValueError('Missing frozen AstroMiNN expert-tower location in registry')
    model_artifact_dir = Path(model_checkpoint).parent
    for path in (validation, manifest, known_manifest, l0, source_checkpoint,
                 initial_checkpoint, current_training, model_artifact_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if manifest.resolve() != known_manifest.resolve():
        raise ValueError('Candidate training manifest not the recorded cumulative L_t')
    if (source_summary.get('cycle') != case.rejected_cycle
            or source_summary.get('strategy') != case.policy
            or str(source_summary.get('trained_checkpoint')) != str(source_checkpoint)):
        raise ValueError('Cycle summary / selected checkpoint mismatch')
    if (int(prior_cycle_summary.get('cycle', -1)) != case.incumbent_cycle
            or str(prior_cycle_summary.get('trained_checkpoint')) != str(initial_checkpoint)):
        raise ValueError('Prior accepted checkpoint/cycle ancestry mismatch')
    if tuple(candidate_eval.get('model_class_names', [])) != CLASSES or tuple(incumbent_eval.get('model_class_names', [])) != CLASSES:
        raise ValueError('Validation class order changed')
    if (train_record['cycle'] != case.rejected_cycle or train_record['seed'] != case.seed
            or train_record['strategy'] != case.policy):
        raise ValueError('Source training cycle mismatch')
    if (Path(train_record['initial_checkpoint']) != initial_checkpoint
            or Path(train_record['checkpoint_path']) != source_checkpoint):
        raise ValueError('Training warm-start ancestry does not match prior incumbent')
    p = train_record['policy']
    if (float(p['class_weight_exponent']) != .85
            or p['new_object_weighting'] != 'per_sample'
            or p['class_weight_source'] != 'training_manifest'
            or p['augment_validation'] is not False
            or p['legacy_seed_sequence'] is not False
            or p['loader_seed'] != case.seed
            or p['max_epochs'] != 50 or p['patience'] != 10
            or p['batch_size'] != 512 or p['validation_batch_size'] != 2048
            or float(p['learning_rate']) != .0001 or p['overwrite'] is not False):
        raise ValueError('Source policy differs from selected Task4 optimization protocol')
    if train_record['initialization'] != 'pretrained':
        raise ValueError('Source AstroMiNN initialization mode differs')
    if check_full_hashes:
        if (sha256(manifest) != train_record['train_manifest_sha256']
                or sha256(validation) != train_record['validation_manifest_sha256']):
            raise ValueError('Training manifest mutated since Task4 checkpoint')
        if sha256(source_checkpoint) != candidate_eval['checkpoint_sha256']:
            raise ValueError('Source candidate checkpoint hash differs from evaluated artifact')
        if sha256(initial_checkpoint) != incumbent_eval['checkpoint_sha256']:
            raise ValueError('Source incumbent checkpoint hash differs from evaluated artifact')
    return VerifiedInputs(
        case, manifest, validation, l0, initial_checkpoint, source_checkpoint,
        current_training, candidate_eval_path, previous_eval_path, model_artifact_dir,
        train_record['train_manifest_sha256'], train_record['validation_manifest_sha256'],
        incumbent_eval['checkpoint_sha256'], candidate_eval['checkpoint_sha256'],
        sha256(current_training), sha256(candidate_eval_path), sha256(previous_eval_path), p)
