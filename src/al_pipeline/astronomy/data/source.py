"""Interfaces for external candidate-data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from al_pipeline.astronomy.data.candidate_pool import CandidatePoolManifest


class CandidateSource(ABC):
    """Base interface for obtaining new unlabeled candidate pools.

    Source-specific configuration, such as observing dates or query
    parameters, should be supplied when constructing the implementation.

    The generic adaptive-learning pipeline only asks a source to materialize
    one named pool into a destination directory.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Stable human-readable name for this external source."""

        ...

    def fetch(
        self,
        *,
        pool_id: str,
        destination: str | Path,
    ) -> CandidatePoolManifest:
        """Fetch one physical candidate pool.

        Implementations should write source-native candidate data beneath
        ``destination`` and return an unlabeled CandidatePoolManifest
        describing those files.
        """

        pool_id = str(pool_id).strip()

        if not pool_id:
            raise ValueError("pool_id must be non-empty.")

        destination = Path(destination)
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest = self._fetch(
            pool_id=pool_id,
            destination=destination,
        )

        if not isinstance(
            manifest,
            CandidatePoolManifest,
        ):
            raise TypeError("Candidate sources must return CandidatePoolManifest.")

        if manifest.pool_id != pool_id:
            raise ValueError(
                f"Candidate source returned pool_id {manifest.pool_id!r}, expected {pool_id!r}."
            )

        return manifest

    @abstractmethod
    def _fetch(
        self,
        *,
        pool_id: str,
        destination: Path,
    ) -> CandidatePoolManifest:
        """Source-specific implementation of candidate retrieval."""

        ...


# TODO: Implement BabamulZTFSource once BOOM/Babamul credentials are
# available and a live ZTF alert has been validated. The source should
# materialize a finite configured batch from the Babamul alert stream and
# return a CandidatePoolManifest.
#
# TODO: Keep CandidateSource synchronous unless continuous online active
# learning is required. Babamul Kafka streams can initially be consumed
# into finite candidate pools.
