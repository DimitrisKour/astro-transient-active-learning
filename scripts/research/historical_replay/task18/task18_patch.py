"""Patch a verified, copied Task11 tree to run AnchorAL only."""
from pathlib import Path
import shutil

def once(s,old,new,name):
    if s.count(old)!=1:
        raise ValueError(f'Task18 patch {name}: expected one anchor, found {s.count(old)}')
    return s.replace(old,new,1)

def patch_source(root,overlay):
    root,overlay=Path(root),Path(overlay)
    newmod='src/al_pipeline/core/advanced_acquisition/anchoral.py'
    dest=root/newmod;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(overlay/newmod,dest)
    changed=[newmod]
    cfg=root/'src/al_pipeline/core/experiments/config.py'
    s=cfg.read_text()
    s=once(s,'    "badge",\n', '    "anchoral",\n    "badge",\n','literal')
    s=once(s,'            "badge", "core_set", "fisher_batch",\n',
           '            "anchoral", "badge", "core_set", "fisher_batch",\n','validator')
    cfg.write_text(s);changed.append(str(cfg.relative_to(root)))
    suite=root/'src/al_pipeline/astronomy/experiments/astrominn.py'
    s=suite.read_text()
    s=once(s,'            "badge", "core_set", "fisher_batch",\n',
           '            "anchoral", "badge", "core_set", "fisher_batch",\n','suite')
    suite.write_text(s);changed.append(str(suite.relative_to(root)))
    sim=root/'src/al_pipeline/astronomy/experiments/simulated.py'
    s=sim.read_text()
    s=once(s,'from al_pipeline.core.advanced_acquisition.badge import BadgeInput, BadgeGradientNormScorer, BadgeSelector\n',
         'from al_pipeline.core.advanced_acquisition.badge import BadgeInput, BadgeGradientNormScorer, BadgeSelector\nfrom al_pipeline.core.advanced_acquisition.anchoral import AnchorALInput, AnchorALSelector\n','import')
    s=once(s,'            if self.config.strategy == "badge":\n                selector = BadgeSelector(',
         '            if self.config.strategy == "anchoral":\n                selector = AnchorALSelector(acquisition_data)\n            elif self.config.strategy == "badge":\n                selector = BadgeSelector(','selector')
    s=once(s,'        if self.config.strategy in {"entropy", "entropy_diversity"}:\n',
         '        if self.config.strategy in {"entropy", "entropy_diversity", "anchoral"}:\n','scorer')
    s=once(s,'        if self.config.strategy == "badge":\n            if inference.embeddings is None:\n',
         '''        if self.config.strategy == "anchoral":
            if inference.embeddings is None:
                raise ValueError("AnchorAL requires object-level embeddings")
            confirmed = self._task11_confirmed_inference(cycle)
            if confirmed.embeddings is None:
                raise ValueError("AnchorAL requires confirmed-set embeddings")
            manifest = self.labeled_sets.training_manifest(cycle - 1)
            labels = {}
            with manifest.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    oid = str(row[self.dataset.object_id_column])
                    label = self.label_mapping.get(row[self.dataset.label_column],
                                                   row[self.dataset.label_column])
                    if oid in labels and labels[oid] != label:
                        raise ValueError("Conflicting confirmed object label")
                    labels[oid] = label
            if set(confirmed.object_ids) != set(labels):
                raise ValueError("Confirmed embeddings and training labels differ")
            return AnchorALInput(
                candidate_ids=tuple(inference.object_ids),
                candidate_embeddings=inference.embeddings,
                labeled_ids=tuple(confirmed.object_ids),
                labeled_embeddings=confirmed.embeddings,
                labeled_labels=tuple(labels[oid] for oid in confirmed.object_ids),
                class_names=tuple(self.class_names), seed=self.config.seed, cycle=cycle,
                anchors_per_class=5, neighbors_per_anchor=200, max_subpool=1419,
            )
        if self.config.strategy == "badge":
            if inference.embeddings is None:
''','scoring-input')
    s=once(s,'                scores = scorer.score(acquisition_data)\n',
           '                scores = scorer.score(inference.predictions if self.config.strategy == "anchoral" else acquisition_data)\n','entropy-input-fix')
    sim.write_text(s);changed.append(str(sim.relative_to(root)))
    runner=root/'scripts/run_astrominn_closed_loop.py'
    s=runner.read_text()
    s=once(s,'            "badge", "core_set", "fisher_batch",\n        ),',
         '            "badge", "core_set", "fisher_batch", "anchoral",\n        ),','choices')
    s=once(s,'    if set(args.strategies) - {"badge", "core_set", "fisher_batch"}:\n        raise ValueError("Task11 runner permits only the three frozen Task11 acquisition policies")\n',
           '    if set(args.strategies) != {"anchoral"}:\n        raise ValueError("Task18 runner permits AnchorAL only")\n','guard')
    runner.write_text(s);changed.append(str(runner.relative_to(root)))
    return changed
