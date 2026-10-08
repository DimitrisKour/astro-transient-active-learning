from pathlib import Path

import pytest

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest, CandidatePoolRecord
from al_pipeline.astronomy.data.source import CandidateSource


class DummySource(CandidateSource):
    @property
    def source_name(self) -> str:
        return "dummy"

    def _fetch(
        self,
        *,
        pool_id: str,
        destination: Path,
    ) -> CandidatePoolManifest:
        sample_path = destination / "sample.npy"
        sample_path.touch()

        return CandidatePoolManifest(
            pool_id=pool_id,
            records=[
                CandidatePoolRecord(
                    survey="TEST",
                    object_id="OBJECT_A",
                    alert=0,
                    filepath=str(sample_path),
                    source_reference="dummy:1",
                )
            ],
        )


class WrongPoolSource(CandidateSource):
    @property
    def source_name(self) -> str:
        return "wrong"

    def _fetch(
        self,
        *,
        pool_id: str,
        destination: Path,
    ) -> CandidatePoolManifest:
        return CandidatePoolManifest(
            pool_id="different_pool",
            records=[
                CandidatePoolRecord(
                    survey="TEST",
                    object_id="OBJECT_A",
                    alert=0,
                    filepath=str(destination / "sample.npy"),
                )
            ],
        )


class InvalidReturnSource(CandidateSource):
    @property
    def source_name(self) -> str:
        return "invalid"

    def _fetch(
        self,
        *,
        pool_id: str,
        destination: Path,
    ):
        return []


def test_source_materializes_pool(tmp_path):
    source = DummySource()

    destination = tmp_path / "new" / "candidate_pool"

    manifest = source.fetch(
        pool_id="test_pool",
        destination=destination,
    )

    assert source.source_name == "dummy"
    assert destination.exists()

    assert manifest.pool_id == "test_pool"

    assert manifest.object_ids == ("OBJECT_A",)

    assert manifest.missing_files() == ()


def test_empty_pool_id_is_rejected(
    tmp_path,
):
    source = DummySource()

    with pytest.raises(
        ValueError,
        match="pool_id must be non-empty",
    ):
        source.fetch(
            pool_id="",
            destination=tmp_path,
        )


def test_wrong_returned_pool_id_is_rejected(
    tmp_path,
):
    source = WrongPoolSource()

    with pytest.raises(
        ValueError,
        match="returned pool_id",
    ):
        source.fetch(
            pool_id="expected_pool",
            destination=tmp_path,
        )


def test_source_must_return_manifest(
    tmp_path,
):
    source = InvalidReturnSource()

    with pytest.raises(
        TypeError,
        match="CandidatePoolManifest",
    ):
        source.fetch(
            pool_id="test_pool",
            destination=tmp_path,
        )
