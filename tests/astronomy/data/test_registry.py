"""Tests for adaptive-learning dataset and profile registry."""

from pathlib import Path

import pytest

from al_pipeline.astronomy.data.registry import DatasetRegistry


def _write_config(
    tmp_path: Path,
    content: str,
) -> Path:
    """Write a temporary TOML registry for one test."""
    path = tmp_path / "dataset_config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_registry_and_resolve_profile(
    tmp_path: Path,
) -> None:
    """A valid profile should resolve model, dataset, and storage."""
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    data_root = tmp_path / "data"

    train.touch()
    val.touch()
    test.touch()
    data_root.mkdir()

    config = _write_config(
        tmp_path,
        f"""
[storage]
persistent_root = "{tmp_path / "persistent"}"
data_root = "{tmp_path / "adaptive_data"}"

[datasets.example]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"

train_manifest = "{train}"
validation_manifest = "{val}"
test_manifest = "{test}"
data_root = "{data_root}"

object_id_column = "obj_id"
sample_file_column = "filepath"
label_column = "label"

feature_stats = "stats.npz"

[profiles.example_profile]
model = "ExampleModel"
dataset = "example"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    resolved = registry.resolve_profile("example_profile")

    assert resolved.profile.name == "example_profile"
    assert resolved.profile.model == "ExampleModel"

    assert resolved.dataset.name == "example"
    assert resolved.dataset.survey == "ZTF"
    assert resolved.dataset.modality == "photometry"
    assert resolved.dataset.selection_unit == "object"

    assert resolved.dataset.train_manifest == train
    assert resolved.dataset.validation_manifest == val
    assert resolved.dataset.test_manifest == test
    assert resolved.dataset.data_root == data_root

    assert resolved.dataset.object_id_column == "obj_id"
    assert resolved.dataset.sample_file_column == "filepath"
    assert resolved.dataset.label_column == "label"

    assert resolved.dataset.options["feature_stats"] == "stats.npz"


def test_dataset_and_profile_names(
    tmp_path: Path,
) -> None:
    """Registry should expose configured names without resolving them."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[datasets.first]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"

[datasets.second]
survey = "Rubin"
modality = "photometry"
selection_unit = "object"

[profiles.first_profile]
model = "ModelA"
dataset = "first"

[profiles.second_profile]
model = "ModelB"
dataset = "second"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    assert registry.dataset_names == (
        "first",
        "second",
    )

    assert registry.profile_names == (
        "first_profile",
        "second_profile",
    )


def test_incomplete_dataset_can_be_registered(
    tmp_path: Path,
) -> None:
    """Known datasets may exist before historical splits are located."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[datasets.spectranet]
survey = "ZTF"
modality = "spectra"
selection_unit = "object"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    dataset = registry.get_dataset("spectranet")

    assert dataset.has_historical_splits is False


def test_validate_paths_accepts_existing_paths(
    tmp_path: Path,
) -> None:
    """Filesystem validation should succeed for complete existing data."""
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    data_root = tmp_path / "data"

    train.touch()
    val.touch()
    test.touch()
    data_root.mkdir()

    config = _write_config(
        tmp_path,
        f"""
[storage]
persistent_root = "{tmp_path / "persistent"}"
data_root = "{tmp_path / "adaptive_data"}"

[datasets.example]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"

train_manifest = "{train}"
validation_manifest = "{val}"
test_manifest = "{test}"
data_root = "{data_root}"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    registry.validate_paths("example")


def test_validate_paths_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Configured paths should be checked only when explicitly requested."""
    config = _write_config(
        tmp_path,
        f"""
[storage]
persistent_root = "{tmp_path / "persistent"}"
data_root = "{tmp_path / "adaptive_data"}"

[datasets.example]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"

train_manifest = "{tmp_path / "missing.csv"}"
validation_manifest = "{tmp_path / "val.csv"}"
test_manifest = "{tmp_path / "test.csv"}"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    with pytest.raises(
        FileNotFoundError,
        match="train_manifest",
    ):
        registry.validate_paths("example")


def test_validate_paths_rejects_incomplete_historical_split(
    tmp_path: Path,
) -> None:
    """Using an incomplete historical dataset should fail explicitly."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[datasets.spectranet]
survey = "ZTF"
modality = "spectra"
selection_unit = "object"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    with pytest.raises(
        ValueError,
        match="does not define complete",
    ):
        registry.validate_paths("spectranet")


def test_unknown_dataset_raises(
    tmp_path: Path,
) -> None:
    """Unknown dataset names should fail explicitly."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[datasets.example]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    with pytest.raises(
        KeyError,
        match="Unknown dataset",
    ):
        registry.get_dataset("missing")


def test_unknown_profile_raises(
    tmp_path: Path,
) -> None:
    """Unknown profile names should fail explicitly."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[datasets.example]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"
""",
    )

    registry = DatasetRegistry.from_toml(config)

    with pytest.raises(
        KeyError,
        match="Unknown profile",
    ):
        registry.get_profile("missing")


def test_profile_cannot_reference_unknown_dataset(
    tmp_path: Path,
) -> None:
    """Profile dataset references must resolve when config is loaded."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[profiles.bad_profile]
model = "ExampleModel"
dataset = "does_not_exist"
""",
    )

    with pytest.raises(
        ValueError,
        match="references unknown dataset",
    ):
        DatasetRegistry.from_toml(config)


def test_dataset_requires_core_metadata(
    tmp_path: Path,
) -> None:
    """Survey, modality, and selection unit are mandatory."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[datasets.incomplete]
survey = "ZTF"
""",
    )

    with pytest.raises(
        ValueError,
        match="missing required field",
    ):
        DatasetRegistry.from_toml(config)


def test_profile_requires_model_and_dataset(
    tmp_path: Path,
) -> None:
    """Profiles must specify both model and dataset."""
    config = _write_config(
        tmp_path,
        """
[storage]
persistent_root = "/persistent"
data_root = "/data"

[datasets.example]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"

[profiles.incomplete]
model = "ExampleModel"
""",
    )

    with pytest.raises(
        ValueError,
        match="missing required field",
    ):
        DatasetRegistry.from_toml(config)


def test_missing_top_level_section_raises(
    tmp_path: Path,
) -> None:
    """Required top-level registry sections should be explicit."""
    config = _write_config(
        tmp_path,
        """
[datasets.example]
survey = "ZTF"
modality = "photometry"
selection_unit = "object"
""",
    )

    with pytest.raises(
        ValueError,
        match="Missing required configuration section",
    ):
        DatasetRegistry.from_toml(config)
