"""Configuration registry for adaptive-learning datasets and profiles.

The registry resolves model/dataset profiles from TOML configuration while
remaining independent of dataset loading, model inference, and the
active-learning loop.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class StorageSpec:
    """Storage owned by adaptive-learning experiments."""

    persistent_root: Path
    data_root: Path


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Description of one historical or future dataset."""

    name: str
    survey: str
    modality: str
    selection_unit: str

    train_manifest: Path | None = None
    validation_manifest: Path | None = None
    test_manifest: Path | None = None
    data_root: Path | None = None

    object_id_column: str | None = None
    sample_file_column: str | None = None
    label_column: str | None = None

    # Dataset-specific configuration that the generic registry does not
    # need to understand, e.g. feature statistics or future Rubin options.
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def has_historical_splits(self) -> bool:
        """Return whether train/validation/test manifests are configured."""
        return all(
            path is not None
            for path in (
                self.train_manifest,
                self.validation_manifest,
                self.test_manifest,
            )
        )


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """Binding between a model and a dataset."""

    name: str
    model: str
    dataset: str

    # Preserve room for future profile-specific options such as checkpoint
    # locations without changing the registry interface.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    """Fully resolved model/dataset/storage configuration."""

    profile: ProfileSpec
    dataset: DatasetSpec
    storage: StorageSpec


class DatasetRegistry:
    """Resolve adaptive-learning datasets and model profiles."""

    def __init__(
        self,
        *,
        storage: StorageSpec,
        datasets: dict[str, DatasetSpec],
        profiles: dict[str, ProfileSpec],
    ) -> None:
        self.storage = storage
        self._datasets = dict(datasets)
        self._profiles = dict(profiles)

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
    ) -> DatasetRegistry:
        """Load a registry from a TOML configuration file."""
        config_path = Path(path)

        with config_path.open("rb") as file:
            config = tomllib.load(file)

        try:
            storage_config = config["storage"]
        except KeyError as exc:
            raise ValueError(f"Missing required configuration section: {exc.args[0]!r}.") from exc

        dataset_configs = config.get("datasets", {})
        profile_configs = config.get("profiles", {})

        storage = StorageSpec(
            persistent_root=Path(storage_config["persistent_root"]),
            data_root=Path(storage_config["data_root"]),
        )

        datasets = {
            name: _parse_dataset(
                name,
                dataset_config,
            )
            for name, dataset_config in dataset_configs.items()
        }

        profiles = {
            name: _parse_profile(
                name,
                profile_config,
            )
            for name, profile_config in profile_configs.items()
        }

        registry = cls(
            storage=storage,
            datasets=datasets,
            profiles=profiles,
        )

        registry._validate_profile_references()

        return registry

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return configured dataset names."""
        return tuple(self._datasets)

    @property
    def profile_names(self) -> tuple[str, ...]:
        """Return configured profile names."""
        return tuple(self._profiles)

    def get_dataset(
        self,
        name: str,
    ) -> DatasetSpec:
        """Return one configured dataset."""
        try:
            return self._datasets[name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset: {name!r}.") from exc

    def get_profile(
        self,
        name: str,
    ) -> ProfileSpec:
        """Return one configured model/dataset profile."""
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown profile: {name!r}.") from exc

    def resolve_profile(
        self,
        name: str,
    ) -> ResolvedProfile:
        """Resolve a profile together with its dataset and storage."""
        profile = self.get_profile(name)
        dataset = self.get_dataset(profile.dataset)

        return ResolvedProfile(
            profile=profile,
            dataset=dataset,
            storage=self.storage,
        )

    def validate_paths(
        self,
        name: str,
        *,
        require_historical_splits: bool = True,
    ) -> None:
        """Validate filesystem paths required by one dataset.

        Path validation is explicit rather than performed when loading the
        registry. This keeps configuration parsing usable in tests and on
        systems where the Delta data are not mounted.
        """
        dataset = self.get_dataset(name)

        if require_historical_splits and not dataset.has_historical_splits:
            raise ValueError(
                f"Dataset {name!r} does not define complete train/validation/test manifests."
            )

        paths = {
            "train_manifest": dataset.train_manifest,
            "validation_manifest": dataset.validation_manifest,
            "test_manifest": dataset.test_manifest,
            "data_root": dataset.data_root,
        }

        for field_name, path in paths.items():
            if path is None:
                continue

            if not path.exists():
                raise FileNotFoundError(f"{field_name} for dataset {name!r} does not exist: {path}")

    def _validate_profile_references(self) -> None:
        """Ensure every profile references a configured dataset."""
        for profile in self._profiles.values():
            if profile.dataset not in self._datasets:
                raise ValueError(
                    f"Profile {profile.name!r} references unknown dataset {profile.dataset!r}."
                )


def _parse_dataset(
    name: str,
    config: dict[str, Any],
) -> DatasetSpec:
    """Create a dataset specification from one TOML table."""
    required = (
        "survey",
        "modality",
        "selection_unit",
    )

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(f"Dataset {name!r} is missing required field(s): {missing}.")

    known_keys = {
        "survey",
        "modality",
        "selection_unit",
        "train_manifest",
        "validation_manifest",
        "test_manifest",
        "data_root",
        "object_id_column",
        "sample_file_column",
        "label_column",
    }

    options = {key: value for key, value in config.items() if key not in known_keys}

    return DatasetSpec(
        name=name,
        survey=str(config["survey"]),
        modality=str(config["modality"]),
        selection_unit=str(config["selection_unit"]),
        train_manifest=_optional_path(config.get("train_manifest")),
        validation_manifest=_optional_path(config.get("validation_manifest")),
        test_manifest=_optional_path(config.get("test_manifest")),
        data_root=_optional_path(config.get("data_root")),
        object_id_column=config.get("object_id_column"),
        sample_file_column=config.get("sample_file_column"),
        label_column=config.get("label_column"),
        options=options,
    )


def _parse_profile(
    name: str,
    config: dict[str, Any],
) -> ProfileSpec:
    """Create a model/dataset profile from one TOML table."""
    required = (
        "model",
        "dataset",
    )

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(f"Profile {name!r} is missing required field(s): {missing}.")

    options = {key: value for key, value in config.items() if key not in {"model", "dataset"}}

    return ProfileSpec(
        name=name,
        model=str(config["model"]),
        dataset=str(config["dataset"]),
        options=options,
    )


def _optional_path(
    value: object,
) -> Path | None:
    """Convert an optional TOML path value to ``Path``."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError("Configured filesystem paths must be strings.")

    return Path(value)
