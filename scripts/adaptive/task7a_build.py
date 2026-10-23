#!/usr/bin/env python3
"""Build isolated Task7A runner plus *frozen-only* AL acquisition code changes."""
from __future__ import annotations
import importlib.util
from pathlib import Path

SUITE='astrominn_closed_loop_warm_temperature_entropy_v1'
BOOT='astrominn_historical_replay_v1'

def replace_one(text:str,old:str,new:str)->str:
    n=text.count(old)
    if n!=1:raise ValueError(f'Upstream anchor requires exactly one occurrence (found {n}): {old[:110]!r}')
    return text.replace(old,new,1)

def runner(repo:Path)->str:
    builder=repo/'scripts/adaptive/task4_build_runner.py'
    orig=repo/'scripts/run_astrominn_historical_replay.py'
    if not builder.is_file() or not orig.is_file():
        raise FileNotFoundError('Task7A requires checked-in Task4 builder and historical-replay runner')
    spec=importlib.util.spec_from_file_location('task7a_original_builder',builder)
    if spec is None or spec.loader is None:raise RuntimeError('Cannot load Task4 runner builder')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    s=mod.build_runner(orig.read_text(encoding='utf-8'))
    s=replace_one(s, '    if args.bootstrap_only and not args.run:\n',
        f'    if args.suite_id != "{SUITE}":\n'
        '        raise ValueError("Task7A must use its own dedicated suite")\n'
        f'    if args.bootstrap_source_suite_id != "{BOOT}":\n'
        '        raise ValueError("Task7A must reuse Experiment B L0")\n'
        '    if args.seed not in (135, 246, 357) or args.strategies != ["entropy"]:\n'
        '        raise ValueError("Task7A supports only three seeds and entropy acquisition")\n'
        '    if args.bootstrap_only:\n'
        '        raise ValueError("Task7A must not retrain any L0 bootstrap")\n'
        '    if args.max_cycles != 9 or not args.complete_pool:\n'
        '        raise ValueError("Task7A has a fixed 9-cycle budget; only cycles 1-5 are queued initially")\n'
        '    if args.bootstrap_only and not args.run:\n')
    s=replace_one(s,'    policy = AstroMiNNViTTrainingConfig(\n',
        '    policy = AstroMiNNViTTrainingConfig(\n'
        '        class_weight_exponent=0.85,\n')
    s=replace_one(s,'    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))',
        '    if (policy.class_weight_exponent != 0.85 or\n'
        '            policy.class_weight_source != "training_manifest" or\n'
        '            policy.new_object_weighting != "per_sample"):\n'
        '        raise AssertionError("Task7A training objective changed")\n'
        '    split_summary = json.loads(replay.summary_path.read_text(encoding="utf-8"))')
    s=replace_one(s, '    if args.extend_budget:\n',
        '    # Enable temperature scaling in the *frozen* AL coordinator.\n'
        '    # Must occur before initialize(), so experiment.json includes the flag.\n'
        '    for experiment in suite.experiments.values():\n'
        '        experiment.temperature_scaling_enabled = True\n'
        '    if args.extend_budget:\n')
    s=replace_one(s,'        "cycle_training_initialization": args.training_initialization,',
        '        "cycle_training_initialization": args.training_initialization,\n'
        '        "task7a_temperature_scaling": "validation_object_nll_before_entropy",\n'
        '        "task7a_weight_exponent": 0.85,\n')
    return s

def simulated(source:str)->str:
    s=replace_one(source,
        '        self.identity_path = self.run_dir / "experiment.json"',
        '        self.identity_path = self.run_dir / "experiment.json"\n'
        '        self.temperature_scaling_enabled = False  # Task7A isolated experiment toggle')
    # Inject into scoring input, not stored inference: saved pool npz stays raw.
    s=replace_one(s, '        return inference.predictions\n',
        '        if self.config.strategy == "entropy" and self.temperature_scaling_enabled:\n'
        '            if inference.predictions.output_type != "logits":\n'
        '                raise ValueError("Temperature scaling requires RAW pool logits")\n'
        '            from al_pipeline.astronomy.experiments.temperature_acquisition import fitted_temperature\n'
        '            previous_evaluation = (self.artifact_dir / f"cycle_{cycle-1:03d}"\n'
        '                                   / "evaluation")\n'
        '            temperature = fitted_temperature(\n'
        '                cycle=cycle,checkpoint=self._inference_checkpoint(cycle),\n'
        '                evaluation_dir=previous_evaluation,\n'
        '                record_path=self.artifact_dir / f"cycle_{cycle:03d}"\n'
        '                            / "temperature_calibration.json",\n'
        '            )\n'
        '            return ClassificationPredictions(\n'
        '                values=inference.predictions.values / temperature, output_type="logits")\n'
        '        return inference.predictions\n')
    s=replace_one(s, '        payload = {\n            "schema_version": 1,',
        '        payload = {\n            "schema_version": 1,\n'
        '            "task7a_temperature_scaling": bool(self.temperature_scaling_enabled),')
    return s

def build_snapshot_files(repo:Path)->tuple[str,str,str]:
    src=(repo/'src/al_pipeline/astronomy/experiments/simulated.py').read_text(encoding='utf-8')
    science=(repo/'scripts/adaptive/task7a_temperature_science.py').read_text(encoding='utf-8')
    return runner(repo),simulated(src),science
