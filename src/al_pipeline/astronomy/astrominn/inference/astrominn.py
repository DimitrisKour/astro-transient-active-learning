"""AstroMiNN inference adapter for adaptive learning."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from al_pipeline.astronomy.astrominn.types import ClassificationInferenceResult
from al_pipeline.core.types import ClassificationPredictions


AstroMiNNInferenceResult = ClassificationInferenceResult


class AstroMiNNPredictor:
    """Run AstroMiNN inference without coupling it to acquisition logic."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cpu",
        embedding_layer: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.embedding_layer = embedding_layer

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        dataloader: DataLoader,
    ) -> AstroMiNNInferenceResult:
        """Return sample-level logits and provenance for a dataloader."""

        object_ids: list[str] = []
        alerts: list[int | None] = []
        logits_batches: list[torch.Tensor] = []
        embedding_batches: list[torch.Tensor] | None = (
            [] if self.embedding_layer is not None else None
        )
        target_indices: list[int] | None = []

        captured_embeddings: list[torch.Tensor] = []
        hook = None

        if self.embedding_layer is not None:

            def capture_embedding(
                _module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
            ) -> None:
                if len(inputs) != 1:
                    raise ValueError("The embedding layer must receive exactly one input.")

                captured_embeddings.append(inputs[0])

            hook = self.embedding_layer.register_forward_pre_hook(capture_embedding)

        try:
            with torch.no_grad():
                for batch in dataloader:
                    metadata = batch["metadata"].to(self.device)
                    image = batch["image"].to(self.device)

                    output = self.model(
                        metadata,
                        image=image,
                    )

                    if not isinstance(output, dict):
                        raise TypeError("AstroMiNN must return a dictionary.")

                    if "logits" not in output:
                        raise KeyError("AstroMiNN output does not contain 'logits'.")

                    logits = output["logits"]

                    if logits.ndim != 2:
                        raise ValueError("AstroMiNN logits must have shape (batch, classes).")

                    batch_size = len(batch["obj_id"])

                    if logits.shape[0] != batch_size:
                        raise ValueError(
                            "AstroMiNN returned a different number "
                            "of predictions than input samples."
                        )

                    if not torch.isfinite(logits).all():
                        raise ValueError("AstroMiNN produced non-finite logits.")

                    logits_batches.append(logits.detach().cpu())

                    if embedding_batches is not None:
                        if len(captured_embeddings) != 1:
                            raise RuntimeError(
                                "The configured embedding layer must be "
                                "called exactly once per inference batch."
                            )

                        embeddings = captured_embeddings.pop()

                        if embeddings.ndim != 2:
                            raise ValueError(
                                "AstroMiNN embeddings must have shape (batch, embedding_dim)."
                            )

                        if embeddings.shape[0] != batch_size:
                            raise ValueError(
                                "AstroMiNN returned a different number of "
                                "embeddings than input samples."
                            )

                        if not torch.isfinite(embeddings).all():
                            raise ValueError("AstroMiNN produced non-finite embeddings.")

                        embedding_batches.append(embeddings.detach().cpu())

                    object_ids.extend(str(obj_id) for obj_id in batch["obj_id"])

                    if "alert" in batch:
                        batch_alerts = batch["alert"]

                        if isinstance(batch_alerts, torch.Tensor):
                            batch_alerts = batch_alerts.tolist()

                        alerts.extend(int(alert) for alert in batch_alerts)

                    else:
                        alerts.extend(None for _ in range(batch_size))

                    if target_indices is not None:
                        if "target_index" not in batch:
                            target_indices = None
                        else:
                            batch_targets = batch["target_index"]

                            if isinstance(batch_targets, torch.Tensor):
                                batch_targets = batch_targets.tolist()

                            target_indices.extend(int(target) for target in batch_targets)
        finally:
            if hook is not None:
                hook.remove()

        if not logits_batches:
            raise ValueError("Cannot predict from an empty dataloader.")

        logits = torch.cat(
            logits_batches,
            dim=0,
        ).numpy()

        return AstroMiNNInferenceResult(
            object_ids=tuple(object_ids),
            alerts=tuple(alerts),
            predictions=ClassificationPredictions(
                values=np.asarray(logits),
                output_type="logits",
            ),
            target_indices=(None if target_indices is None else tuple(target_indices)),
            embeddings=(
                None if embedding_batches is None else torch.cat(embedding_batches, dim=0).numpy()
            ),
        )
