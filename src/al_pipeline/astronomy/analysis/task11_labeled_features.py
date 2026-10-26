"""Confirmed-training-manifest inference under the same frozen AstroMiNN model.

This is *not* inference on candidate oracle labels: only already-confirmed L_t
is used as an anchor set. Latest-alert aggregation mirrors U_t inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from al_pipeline.astronomy.astrominn.training import AstroMiNNViTManifestDataset
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import load_astrominn_vit_model
from al_pipeline.astronomy.astrominn.inference.astrominn import AstroMiNNPredictor
from al_pipeline.astronomy.astrominn.inference.aggregation import aggregate_object_predictions
from al_pipeline.astronomy.astrominn.types import ClassificationInferenceResult


def predict_confirmed_objects(*, manifest_path, dataset, checkpoint_path, artifact_dir,
                              device, batch_size, num_workers, legacy_code_dir):
    ds = AstroMiNNViTManifestDataset(
        manifest_path=manifest_path, dataset=dataset, artifact_dir=artifact_dir,
        augment=False, cache_in_memory=False)
    model = load_astrominn_vit_model(
        legacy_code_dir=legacy_code_dir, artifact_dir=artifact_dir,
        vit_checkpoint=checkpoint_path)
    pred = AstroMiNNPredictor(
        model, device=device, embedding_layer=model.vit_tower.fc_out,
    ).predict(DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=num_workers))
    if pred.embeddings is None:
        raise RuntimeError('Missing confirmed-set classifier-input embeddings')
    if len(set(pred.object_ids)) == len(pred.object_ids):
        return ClassificationInferenceResult(
            object_ids=pred.object_ids, alerts=pred.alerts, predictions=pred.predictions,
            embeddings=pred.embeddings)
    aggregated = aggregate_object_predictions(
        object_ids=pred.object_ids, alerts=pred.alerts,
        predictions=pred.predictions, method='latest')
    if aggregated.source_indices is None:
        raise RuntimeError('Unable to associate latest-alert representation')
    ix = np.asarray(aggregated.source_indices,dtype=np.int64)
    alerts = tuple(pred.alerts[int(i)] for i in ix)
    result = ClassificationInferenceResult(
        object_ids=aggregated.object_ids, alerts=alerts,
        predictions=aggregated.predictions,embeddings=pred.embeddings[ix])
    if len(set(result.object_ids)) != len(result.object_ids):
        raise RuntimeError('Confirmed-set object aggregation failed')
    return result
