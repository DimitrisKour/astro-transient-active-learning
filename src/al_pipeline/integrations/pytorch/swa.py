"""Opt-in PyTorch stochastic weight averaging (SWA) for training backends.

This module lives OUTSIDE ``al_pipeline.core``: the core architecture
forbids torch imports even in lazily imported core modules. Only concrete
PyTorch training backends need import this optional adapter. The AL loop, training requests, checkpoint
selection, and acceptance machinery are unchanged.

SWA is an averaged point estimate, not an ensemble or uncertainty posterior.

TODO(SWAG): Consider a SEPARATE Stochastic Weight Averaging-Gaussian adapter
for covariance/moment tracking, posterior weight sampling and uncertainty
scores. Design its memory budget, BatchNorm refresh per sampled model,
calibration and ensemble interfaces before implementing it. Do not silently
interpret SWA predictive probabilities as Bayesian uncertainty.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from al_pipeline.core.training.checkpoint_selection import CheckpointTrainingIdentity


@dataclass(frozen=True, slots=True)
class SWAConfig:
    """An explicit, disabled-by-default SWA schedule.

    start_epoch is 1-indexed; an average is added at ``start_epoch`` and
    every ``update_every`` completed epochs thereafter. If swa_lr is set,
    ``SWALR`` replaces the caller's base scheduler starting with that epoch.
    With swa_lr=None, the base scheduler continues unchanged.
    """

    enabled: bool = False
    start_epoch: int = 10
    update_every: int = 1
    swa_lr: float | None = None
    anneal_epochs: int = 10
    anneal_strategy: Literal["cos", "linear"] = "cos"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if type(self.start_epoch) is not int or self.start_epoch < 1:
            raise ValueError("start_epoch must be a positive integer")
        if type(self.update_every) is not int or self.update_every < 1:
            raise ValueError("update_every must be a positive integer")
        if type(self.anneal_epochs) is not int or self.anneal_epochs < 0:
            raise ValueError("anneal_epochs must be a nonnegative integer")
        if self.anneal_strategy not in {"cos", "linear"}:
            raise ValueError("anneal_strategy must be 'cos' or 'linear'")
        if self.swa_lr is not None:
            if isinstance(self.swa_lr, (bool, str, bytes)):
                raise TypeError("swa_lr must be a finite positive number")
            try:
                lr = float(self.swa_lr)
            except (TypeError, ValueError) as exc:
                raise TypeError("swa_lr must be a finite positive number") from exc
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError("swa_lr must be finite and positive")
            object.__setattr__(self, "swa_lr", lr)


@dataclass(frozen=True, slots=True)
class SWAEpochUpdate:
    epoch: int
    swa_active: bool
    averaged: bool
    n_averaged: int
    scheduler_used: Literal["swa", "base", "none"]


class SWATracker:
    """Accumulate snapshots from one model during a single training fit.

    Call ``end_epoch`` once AFTER optimizer updates have completed. Pass the
    base LR scheduler if one exists. The tracker steps exactly one scheduler:
    PyTorch SWALR during SWA phase (when configured), otherwise the base
    scheduler. The caller must not separately step those schedulers.

    Normal training epochs, gradient/optimizer state, early stopping and
    validation remain under the caller's control. SWA never selects a model or
    changes acquisition/labels. All checkpoints are independent candidates.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: SWAConfig = SWAConfig(),
        optimizer: torch.optim.Optimizer | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if config.swa_lr is not None and config.enabled and optimizer is None:
            raise ValueError("An optimizer is required when swa_lr is configured")
        self.model = model
        self.config = config
        self.averaged_model = (
            AveragedModel(model, device=device, use_buffers=False)
            if config.enabled else None
        )
        self.swa_scheduler = (
            SWALR(
                optimizer,
                swa_lr=config.swa_lr,
                anneal_epochs=config.anneal_epochs,
                anneal_strategy=config.anneal_strategy,
            )
            if config.enabled and config.swa_lr is not None else None
        )
        self.averaged_epochs: list[int] = []
        self.last_epoch = 0
        self._bn_refreshed = False

    @property
    def n_averaged(self) -> int:
        if self.averaged_model is None:
            return 0
        return int(self.averaged_model.n_averaged.item())

    @property
    def ready(self) -> bool:
        return self.n_averaged > 0

    def end_epoch(
        self,
        epoch: int,
        *,
        base_scheduler: Any | None = None,
    ) -> SWAEpochUpdate:
        """Update average when eligible and step ONE learning-rate scheduler.

        Call only after successful optimizer steps for the epoch. If training
        stops before start_epoch, no averaged model is available; ordinary
        checkpointing and evaluation are unaffected.
        """
        if type(epoch) is not int or epoch <= self.last_epoch:
            raise ValueError("epoch must be an integer strictly after last_epoch")
        if epoch != self.last_epoch + 1:
            raise ValueError("epoch must be consecutive; load run state when resuming")
        active = self.config.enabled and epoch >= self.config.start_epoch
        average = active and (epoch - self.config.start_epoch) % self.config.update_every == 0
        if average:
            assert self.averaged_model is not None
            self.averaged_model.update_parameters(self.model)
            self.averaged_epochs.append(epoch)
            self._bn_refreshed = False

        if active and self.swa_scheduler is not None:
            self.swa_scheduler.step()
            stepped: Literal["swa", "base", "none"] = "swa"
        elif base_scheduler is not None:
            base_scheduler.step()
            stepped = "base"
        else:
            stepped = "none"
        self.last_epoch = epoch
        return SWAEpochUpdate(
            epoch=epoch, swa_active=bool(active), averaged=bool(average),
            n_averaged=self.n_averaged, scheduler_used=stepped,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return averaging state; resume also requires the caller's model,
        optimizer and base-scheduler states from the SAME completed epoch.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": asdict(self.config),
            "last_epoch": self.last_epoch,
            "averaged_epochs": list(self.averaged_epochs),
            "averaged_model": (
                copy.deepcopy(self.averaged_model.state_dict())
                if self.averaged_model is not None else None
            ),
            "swa_scheduler": (
                copy.deepcopy(self.swa_scheduler.state_dict())
                if self.swa_scheduler is not None else None
            ),
            "bn_refreshed": self._bn_refreshed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an accumulator after external training state has been restored.

        Rejects incompatible config, invalid epoch sequences and missing
        scheduler state before modifying the averaged model.
        """
        if not isinstance(state, Mapping) or state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported SWA tracker state schema")
        if state.get("config") != asdict(self.config):
            raise ValueError("SWA configuration does not match saved state")
        last = state.get("last_epoch")
        if type(last) is not int or last < 0:
            raise ValueError("Invalid SWA last_epoch")
        epochs = state.get("averaged_epochs")
        if not isinstance(epochs, list) or any(type(e) is not int for e in epochs):
            raise ValueError("Invalid SWA averaged_epochs")
        expected = (
            list(range(self.config.start_epoch, last + 1, self.config.update_every))
            if self.config.enabled and last >= self.config.start_epoch else []
        )
        if epochs != expected:
            raise ValueError("SWA averaged_epochs do not match the configured schedule")
        swa_state = state.get("averaged_model")
        scheduler_state = state.get("swa_scheduler")
        if (self.averaged_model is None) != (swa_state is None):
            raise ValueError("Missing or unexpected SWA averaged model state")
        if (self.swa_scheduler is None) != (scheduler_state is None):
            raise ValueError("Missing or unexpected SWALR scheduler state")
        if self.averaged_model is not None:
            if not isinstance(swa_state, Mapping) or "n_averaged" not in swa_state:
                raise ValueError("Invalid SWA model state")
            n = swa_state["n_averaged"]
            if not isinstance(n, torch.Tensor) or n.numel() != 1 or int(n.item()) != len(epochs):
                raise ValueError("SWA n_averaged inconsistent with saved epochs")
        refreshed = state.get("bn_refreshed")
        if type(refreshed) is not bool:
            raise ValueError("Invalid SWA bn_refreshed flag")
        if self.averaged_model is not None:
            self.averaged_model.load_state_dict(swa_state, strict=True)
        if self.swa_scheduler is not None:
            self.swa_scheduler.load_state_dict(scheduler_state)
        self.averaged_epochs = list(epochs)
        self.last_epoch = last
        self._bn_refreshed = refreshed

    def _swa_model_for_export(
        self,
        *,
        bn_loader: Any | None = None,
        bn_refresh: Callable[[torch.nn.Module], None] | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[torch.nn.Module, bool]:
        if not self.ready or self.averaged_model is None:
            raise RuntimeError("No SWA weights available; complete an eligible epoch first")
        if bn_loader is not None and bn_refresh is not None:
            raise ValueError("Specify either bn_loader or bn_refresh, not both")
        model = self.averaged_model.module
        needs_bn = any(
            isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()
        )
        if needs_bn and not self._bn_refreshed:
            if bn_loader is None and bn_refresh is None:
                raise ValueError("SWA model contains BatchNorm: provide bn_loader or bn_refresh")
        if bn_loader is not None:
            # torch.update_bn feeds one tensor to forward. For multimodal
            # models such as AstroMiNN(metadata, image), supply bn_refresh
            # instead, with a backend-specific multi-input forward callback.
            update_bn(bn_loader, self.averaged_model, device=device)
            self._bn_refreshed = True
        elif bn_refresh is not None:
            bn_refresh(model)
            self._bn_refreshed = True
        return model, bool(needs_bn and self._bn_refreshed)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        identity: CheckpointTrainingIdentity,
        bn_loader: Any | None = None,
        bn_refresh: Callable[[torch.nn.Module], None] | None = None,
        device: str | torch.device | None = None,
    ) -> Path:
        """Export a separate, immutable SWA candidate checkpoint atomically.

        Payload contains a plain model state_dict (no AveragedModel prefix),
        B's fit identity, and averaging schedule provenance. It intentionally
        does not overwrite any best-loss/best-F1 checkpoint. The astronomy
        backend will adapt this payload to its own checkpoint format later.

        This is an SWA *evaluation candidate*, not an automatic acceptance.
        """
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        model, bn_refreshed = self._swa_model_for_export(
            bn_loader=bn_loader, bn_refresh=bn_refresh, device=device,
        )
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "checkpoint_type": "swa",
            "identity": asdict(identity),
            "swa": {
                "config": asdict(self.config),
                "averaged_epochs": list(self.averaged_epochs),
                "n_averaged": self.n_averaged,
                "last_epoch": self.last_epoch,
                "batchnorm_refreshed": bn_refreshed,
            },
            "model_state": {
                key: (value.detach().cpu().clone() if isinstance(value, torch.Tensor)
                      else copy.deepcopy(value))
                for key, value in model.state_dict().items()
            },
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".pending", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # Single-writer invariant, as for Package B's checkpoint keeper.
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination


def load_swa_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    expected_identity: CheckpointTrainingIdentity,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load exported SWA weights into a compatible model with strict checks.

    Returns SWA provenance. This is evaluation/deployment restoration, NOT a
    training-trajectory resume; resume needs SWATracker.state_dict AND the
    caller's model/optimizer/scheduler states.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("schema_version") != SWATracker.SCHEMA_VERSION:
        raise ValueError("Unsupported SWA checkpoint format")
    if payload.get("checkpoint_type") != "swa":
        raise ValueError("Not an SWA checkpoint")
    if payload.get("identity") != asdict(expected_identity):
        raise ValueError("SWA checkpoint training identity mismatch")
    provenance = payload.get("swa")
    if not isinstance(provenance, dict) or provenance.get("n_averaged", 0) <= 0:
        raise ValueError("Missing or invalid SWA provenance")
    model.load_state_dict(payload["model_state"], strict=True)
    return provenance


__all__ = ["SWAConfig", "SWAEpochUpdate", "SWATracker", "load_swa_checkpoint"]
