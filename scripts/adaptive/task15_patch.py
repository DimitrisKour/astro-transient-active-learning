"""Strict patch against an independently verified copy of frozen Task11 source."""
from pathlib import Path
import shutil

def once(text,old,new,label):
    count=text.count(old)
    if count!=1:raise ValueError(f'Task15 patch failed {label}: {count} anchors')
    return text.replace(old,new,1)

def patch_source(root,overlay):
    root=Path(root);overlay=Path(overlay)
    target=root/'src/al_pipeline/core/advanced_acquisition/direct.py'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(overlay/'src/al_pipeline/core/advanced_acquisition/direct.py',target)
    p=root/'src/al_pipeline/core/experiments/config.py';text=p.read_text()
    text=once(text,'StrategyName: TypeAlias = Literal[\n','StrategyName: TypeAlias = Literal[\n    "direct",\n','config-type')
    text=once(text,'        if self.strategy not in {\n','        if self.strategy not in {\n            "direct",\n','config-guard')
    p.write_text(text)
    p=root/'src/al_pipeline/astronomy/experiments/astrominn.py';text=p.read_text()
    text=once(text,'            "badge", "core_set", "fisher_batch",\n',
                    '            "badge", "core_set", "fisher_batch", "direct",\n','suite-allowed')
    p.write_text(text)
    p=root/'scripts/run_astrominn_closed_loop.py';text=p.read_text()
    text=once(text,'            "badge", "core_set", "fisher_batch",\n',
                    '            "badge", "core_set", "fisher_batch", "direct",\n','runner-choices')
    text=once(text,'    if set(args.strategies) - {"badge", "core_set", "fisher_batch"}:\n        raise ValueError("Task11 runner permits only the three frozen Task11 acquisition policies")\n',
                    '    if set(args.strategies) - {"direct"}:\n        raise ValueError("Task15 runner permits only the frozen DIRECT acquisition policy")\n', 'runner-lock')
    p.write_text(text)
    p=root/'src/al_pipeline/astronomy/experiments/simulated.py';text=p.read_text()
    text=once(text,'from al_pipeline.core.advanced_acquisition.fisher_batch import FisherBatchInput, FisherBatchScorer, FisherBatchSelector\n',
                    'from al_pipeline.core.advanced_acquisition.fisher_batch import FisherBatchInput, FisherBatchScorer, FisherBatchSelector\n'
                    'from al_pipeline.core.advanced_acquisition.direct import DirectInput, DirectSelector\n', 'sim-import')
    text=once(text,'        if path.exists():\n            selection = _load_selection(path)\n        else:\n            scorer = self._scorer(cycle)\n',
                    '        if self.config.strategy == "direct":\n            return self._task15_select_or_resume(inference=inference, budget=budget, cycle=cycle, path=path)\n        if path.exists():\n            selection = _load_selection(path)\n        else:\n            scorer = self._scorer(cycle)\n', 'sim-selection-route')
    helper='''    def _task15_confirmed_labels(self, cycle: int) -> dict[str, str]:
        """Deduplicated, confirmed labels from previous-cycle training only."""
        path = self.labeled_sets.training_manifest(cycle-1)
        labels = {}
        with path.open(newline='',encoding='utf-8') as stream:
            for record in csv.DictReader(stream):
                oid=record[self.dataset.object_id_column]
                raw=record[self.dataset.label_column]
                label=self.label_mapping.get(raw,raw)
                if label not in self.class_names:
                    raise ValueError(f'Unexpected known DIRECT class: {raw!r}')
                if oid in labels and labels[oid]!=label:
                    raise ValueError('Contradictory confirmed object labels')
                labels[oid]=label
        return labels

    def _task15_select_or_resume(self, *, inference, budget, cycle, path):
        """Perform resumable *interactive* oracle subrounds, before retraining.

        Journal persistence ensures a crashed job replays identical queries
        and labels. No unknown label enters the selector before its query.
        The regular experiment's later _labels_or_resume is consistent with
        this journal but cannot create new labels on its own.
        """
        if path.exists():
            selection=_load_selection(path)
            if len(selection)!=budget or not set(selection.ids).issubset(inference.object_ids):
                raise ValueError('Saved DIRECT selection identity mismatch')
            return selection
        anchor=self._task11_confirmed_inference(cycle)
        labels=self._task15_confirmed_labels(cycle)
        if set(anchor.object_ids)!=set(labels):
            raise ValueError('DIRECT confirmed inference disagrees with manifest')
        data=DirectInput(
            candidate_ids=tuple(inference.object_ids),
            probabilities=inference.predictions.as_probabilities(),
            confirmed_ids=tuple(anchor.object_ids),
            confirmed_probabilities=anchor.predictions.as_probabilities(),
            confirmed_labels=tuple(labels[x] for x in anchor.object_ids),
            class_names=tuple(self.class_names))
        journal=path.with_name('direct_oracle_journal.json')
        records=json.loads(journal.read_text()) if journal.is_file() else []
        if not isinstance(records,list):raise ValueError('Invalid DIRECT oracle journal')
        cursor=0
        from al_pipeline.core.types import SelectionBatch
        oracle=ExistingLabelsOracle(self.hidden_labels)
        def query(ids, *, phase, class_name, subround):
            nonlocal cursor
            if cursor<len(records):
                entry=records[cursor]
                if entry['ids']!=list(ids) or entry['phase']!=phase or entry['class']!=class_name or entry['subround']!=subround:
                    raise RuntimeError('DIRECT deterministic journal replay diverged')
                out=entry['labels']
            else:
                request=SelectionBatch(ids=list(ids),indices=np.arange(len(ids)),scores=np.zeros(len(ids)))
                observed=oracle.query(request)
                out=[str(x) for x in observed.labels]
                records.append({'phase':phase,'class':class_name,'subround':subround,
                                'ids':list(ids),'labels':out})
                _json_dump_atomic(records,journal)
            cursor+=1
            return tuple(out)
        seed=int(np.random.SeedSequence([self.config.seed,cycle,15015]).generate_state(1)[0])
        selection=DirectSelector(data,parallel_batch=5,seed=seed).select(budget=budget,query=query)
        if cursor!=len(records):raise RuntimeError('Unused journal queries detected')
        if len(selection)!=budget or len(set(selection.ids))!=budget:
            raise RuntimeError('DIRECT budget or uniqueness violated')
        revealed={oid:label for entry in records for oid,label in zip(entry['ids'],entry['labels'],strict=True)}
        if set(revealed)!=set(selection.ids):raise RuntimeError('DIRECT journal and selection differ')
        selection.metadata['oracle_journal']='direct_oracle_journal.json'
        selection.metadata['oracle_query_count']=len(records)
        _save_selection(selection,path)
        return selection

'''
    text=once(text,'    def _labels_or_resume(\n', helper+'    def _labels_or_resume(\n', 'sim-oracle-helpers')
    text=once(text,'        if path.exists():\n            labeled = _load_labeled(path)\n        else:\n            labeled = ExistingLabelsOracle(self.hidden_labels).query(selection)\n            _save_labeled(labeled, path)\n',
'''        if path.exists():
            labeled = _load_labeled(path)
        elif self.config.strategy == "direct":
            journal=path.with_name('direct_oracle_journal.json')
            if not journal.is_file():raise RuntimeError('DIRECT journal absent')
            records=json.loads(journal.read_text())
            queried={oid:label for record in records for oid,label in zip(record['ids'],record['labels'],strict=True)}
            if set(queried)!=set(selection.ids) or len(queried)!=len(selection):
                raise RuntimeError('DIRECT selection differs from queried labels')
            from al_pipeline.core.types import LabeledBatch
            labeled=LabeledBatch(ids=list(selection.ids), labels=np.asarray([queried[oid] for oid in selection.ids]))
            _save_labeled(labeled,path)
        else:
            labeled = ExistingLabelsOracle(self.hidden_labels).query(selection)
            _save_labeled(labeled, path)
''','sim-postlabels')
    p.write_text(text)
