import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.astrominn.inference.astrominn import AstroMiNNPredictor
from al_pipeline.astronomy.astrominn.inference.astrominn_vit import AstroMiNNViTPoolDataset


class DummyDataset(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {
            "obj_id": f"object-{index}",
            "alert": index + 1,
            "metadata": torch.tensor([float(index), 1.0]),
            "image": torch.ones(
                3,
                4,
                4,
            )
            * index,
        }


class DummyAstroMiNN(nn.Module):
    def forward(
        self,
        metadata,
        image=None,
    ):
        batch_size = metadata.shape[0]

        logits = torch.arange(
            batch_size * 5,
            dtype=torch.float32,
            device=metadata.device,
        ).reshape(batch_size, 5)

        return {
            "logits": logits,
        }


class DummyEmbeddingAstroMiNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, 4, bias=False)
        self.classifier = nn.Linear(4, 5)

    def forward(self, metadata, image=None):
        embedding = self.embedding(metadata)
        return {"logits": self.classifier(embedding)}


def test_predictor_returns_logits_and_object_ids():
    loader = DataLoader(
        DummyDataset(),
        batch_size=2,
        shuffle=False,
    )

    predictor = AstroMiNNPredictor(
        DummyAstroMiNN(),
        device="cpu",
    )

    result = predictor.predict(loader)

    assert result.object_ids == (
        "object-0",
        "object-1",
        "object-2",
    )

    assert result.predictions.output_type == "logits"

    assert result.predictions.values.shape == (
        3,
        5,
    )

    assert result.alerts == (
        1,
        2,
        3,
    )

    assert np.isfinite(result.predictions.values).all()


def test_predictor_probabilities_are_valid():
    loader = DataLoader(
        DummyDataset(),
        batch_size=2,
        shuffle=False,
    )

    predictor = AstroMiNNPredictor(
        DummyAstroMiNN(),
    )

    result = predictor.predict(loader)

    probabilities = result.predictions.as_probabilities()

    assert probabilities.shape == (
        3,
        5,
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_predictor_captures_final_layer_input_embeddings():
    loader = DataLoader(
        DummyDataset(),
        batch_size=2,
        shuffle=False,
    )
    model = DummyEmbeddingAstroMiNN()
    predictor = AstroMiNNPredictor(
        model,
        embedding_layer=model.classifier,
    )

    result = predictor.predict(loader)

    metadata = torch.stack([DummyDataset()[index]["metadata"] for index in range(3)])
    expected = model.embedding(metadata).detach().numpy()

    assert result.embeddings is not None
    assert result.embeddings.shape == (3, 4)
    assert np.allclose(result.embeddings, expected)

    # Hooks must not remain attached after inference completes.
    assert not model.classifier._forward_pre_hooks


def test_predictor_rejects_empty_loader():
    class EmptyDataset(Dataset):
        def __len__(self):
            return 0

        def __getitem__(self, index):
            raise IndexError

    predictor = AstroMiNNPredictor(
        DummyAstroMiNN(),
    )

    loader = DataLoader(
        EmptyDataset(),
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        predictor.predict(loader)


def test_pool_dataset_can_restrict_current_unlabeled_ids(
    tmp_path: Path,
) -> None:
    pool_path = tmp_path / "pool"
    samples = pool_path / "samples"
    samples.mkdir(parents=True)
    records = []

    for index, object_id in enumerate(("a", "b")):
        sample_path = samples / f"{object_id}.npy"
        np.save(
            sample_path,
            {
                "obj_id": object_id,
                "alerte": index + 1,
                "metadata": np.zeros(34, dtype=np.float32),
                "image": np.ones((3, 63, 63), dtype=np.float32),
            },
            allow_pickle=True,
        )
        records.append(
            CandidatePoolRecord(
                survey="ZTF",
                object_id=object_id,
                alert=index + 1,
                filepath=str(sample_path),
            )
        )

    CandidatePoolManifest(
        pool_id="test",
        records=records,
    ).to_csv(pool_path / "candidate_pool.csv")
    (pool_path / "ingestion_report.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "astroMiNN_config.json").write_text(
        json.dumps({"patch_size": [45, 5], "image_norm": "median"}),
        encoding="utf-8",
    )

    dataset = AstroMiNNViTPoolDataset(
        pool_path,
        artifact_dir=artifact_dir,
        object_ids=("b",),
    )

    assert len(dataset) == 1
    assert dataset[0]["obj_id"] == "b"

    with pytest.raises(ValueError, match="not present"):
        AstroMiNNViTPoolDataset(
            pool_path,
            artifact_dir=artifact_dir,
            object_ids=("missing",),
        )
