"""Provider-neutral ZTF alert adaptation for historical AstroMiNN."""

from __future__ import annotations

import gzip
import io
import warnings

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

from al_pipeline.astronomy.ztf.photometry import ZTFPhotometryFeatureBuilder
from al_pipeline.astronomy.ztf.preprocessing import (
    PreparedAstroMiNNSample,
    ZTFAstroMiNNPreprocessor,
)


class ZTFPacketError(ValueError):
    """Raised when a ZTF alert cannot be safely converted."""


class ZTFAstroMiNNAlertAdapter:
    """Translate one reconstructed ZTF alert into AstroMiNN input.

    The input may originate from any provider, provided it contains:

    - objectId
    - candidate
    - prv_candidates
    - cutoutScience
    - cutoutTemplate
    - cutoutDifference

    Provider-specific transport and storage details must be resolved before
    calling this adapter.
    """

    def __init__(
        self,
        *,
        max_alert: int = 10,
    ) -> None:
        if max_alert < 0:
            raise ValueError("max_alert must be >= 0.")

        self.max_alert = int(max_alert)

        self.photometry_builder = ZTFPhotometryFeatureBuilder()

        self.preprocessor = ZTFAstroMiNNPreprocessor()

    def alert_number(
        self,
        packet: Mapping[str, Any],
    ) -> int:
        """Return the historical zero-based AstroMiNN alert index.

        BOOM auxiliary histories are object-level and may contain detections
        occurring after the current alert. Only detections strictly before
        the current candidate JD are therefore used.

        If a provider supplies ``nalerthist``, it is used only as an
        additional consistency check.
        """

        candidate = self._candidate(packet)

        current_jd = float(
            self._required(
                candidate,
                "jd",
            )
        )

        previous = self._usable_previous(
            packet,
            before_jd=current_jd,
        )

        alert_number = len(previous)

        declared = packet.get("nalerthist")

        if declared is not None:
            declared_alert = int(declared) - 1

            if declared_alert < 0:
                raise ZTFPacketError("nalerthist must be >= 1.")

            if declared_alert != alert_number:
                raise ZTFPacketError(
                    "nalerthist is inconsistent "
                    "with usable previous detections: "
                    f"declared={declared_alert}, "
                    f"derived={alert_number}."
                )

        return alert_number

    def is_historical_compatible(
        self,
        packet: Mapping[str, Any],
    ) -> bool:
        """Whether this alert lies in the historical <= max_alert regime."""

        return self.alert_number(packet) <= self.max_alert

    def build_photometry(
        self,
        packet: Mapping[str, Any],
    ) -> list[dict[str, float | int | None]]:
        """Build detection history through the current alert."""

        candidate = self._candidate(packet)

        current_jd = float(
            self._required(
                candidate,
                "jd",
            )
        )

        current = {
            "jd": current_jd,
            "magpsf": float(
                self._required(
                    candidate,
                    "magpsf",
                )
            ),
            "fid": int(
                self._required(
                    candidate,
                    "fid",
                )
            ),
        }

        rows = self._usable_previous(
            packet,
            before_jd=current_jd,
        )

        rows.append(current)

        rows.sort(key=lambda row: row["jd"])

        return rows

    def build_candidate_metadata(
        self,
        packet: Mapping[str, Any],
    ) -> dict[str, float]:
        """Translate native ZTF candidate values to canonical names."""

        candidate = self._candidate(packet)

        names = (
            "sgscore1",
            "sgscore2",
            "distpsnr1",
            "distpsnr2",
            "ra",
            "dec",
            "nmtchps",
            "sharpnr",
            "scorr",
            "sky",
            "diffmaglim",
            "ndethist",
            "ncovhist",
            "sigmapsf",
            "chinr",
            "magpsf",
            "classtar",
        )

        metadata = {
            name: float(
                self._required(
                    candidate,
                    name,
                )
            )
            for name in names
        }

        metadata["filter_id"] = float(
            self._required(
                candidate,
                "fid",
            )
        )

        # Historical AstroMiNN preprocessing used nnondet.
        metadata["nnondet"] = metadata["ncovhist"] - metadata["ndethist"]

        return metadata

    def build_image(
        self,
        packet: Mapping[str, Any],
    ) -> np.ndarray:
        """Decode the three compressed ZTF FITS cutouts."""

        channels = [
            self._decode_cutout(
                packet,
                "cutoutScience",
            ),
            self._decode_cutout(
                packet,
                "cutoutTemplate",
            ),
            self._decode_cutout(
                packet,
                "cutoutDifference",
            ),
        ]

        image = np.stack(
            channels,
            axis=-1,
        )

        if image.shape != (63, 63, 3):
            raise ZTFPacketError(
                f"Expected decoded ZTF cutouts to produce shape (63, 63, 3), got {image.shape}."
            )

        return image

    def prepare(
        self,
        packet: Mapping[str, Any],
    ) -> PreparedAstroMiNNSample:
        """Prepare one unlabeled ZTF alert for historical AstroMiNN."""

        object_id = packet.get("objectId")

        if not object_id:
            raise ZTFPacketError("ZTF packet is missing objectId.")

        alert_number = self.alert_number(packet)

        if alert_number > self.max_alert:
            raise ZTFPacketError(
                "ZTF alert lies outside the "
                "historical AstroMiNN alert range: "
                f"{alert_number} > {self.max_alert}."
            )

        metadata = self.build_candidate_metadata(packet)

        metadata = self.photometry_builder.augment_metadata(
            metadata,
            self.build_photometry(packet),
        )

        return self.preprocessor.preprocess(
            object_id=str(object_id),
            alert_number=alert_number,
            images=self.build_image(packet),
            metadata=metadata,
        )

    @staticmethod
    def _candidate(
        packet: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        candidate = packet.get("candidate")

        if not isinstance(
            candidate,
            Mapping,
        ):
            raise ZTFPacketError("ZTF packet is missing candidate.")

        return candidate

    @staticmethod
    def _required(
        mapping: Mapping[str, Any],
        key: str,
    ) -> Any:
        value = mapping.get(key)

        if value is None:
            raise ZTFPacketError(f"Required ZTF field {key!r} is missing.")

        return value

    @staticmethod
    def _usable_previous(
        packet: Mapping[str, Any],
        *,
        before_jd: float,
    ) -> list[dict[str, float | int | None]]:
        """Return usable detections strictly before the current alert."""

        previous = packet.get("prv_candidates") or []

        rows: list[dict[str, float | int | None]] = []

        for candidate in previous:
            if not isinstance(
                candidate,
                Mapping,
            ):
                continue

            if candidate.get("jd") is None or candidate.get("magpsf") is None:
                continue

            try:
                jd = float(candidate["jd"])

                magpsf = float(candidate["magpsf"])

            except (
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(jd) or not np.isfinite(magpsf):
                continue

            # Critical for BOOM: auxiliary histories can contain
            # detections occurring after this particular alert.
            if jd >= before_jd:
                continue

            raw_fid = candidate.get("fid")

            try:
                fid = None if raw_fid is None else int(raw_fid)
            except (
                TypeError,
                ValueError,
            ):
                fid = None

            rows.append(
                {
                    "jd": jd,
                    "magpsf": magpsf,
                    "fid": fid,
                }
            )

        return rows

    @classmethod
    def _decode_cutout(
        cls,
        packet: Mapping[str, Any],
        name: str,
    ) -> np.ndarray:
        """Decode compressed cutouts from supported ZTF packet representations."""

        if name not in packet:
            raise ZTFPacketError(f"ZTF packet is missing {name}.")

        stamp_data = cls._stamp_bytes(
            packet[name],
            name=name,
        )

        try:
            content = gzip.decompress(stamp_data)

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    VerifyWarning,
                )

                with fits.open(
                    io.BytesIO(content),
                    memmap=False,
                ) as hdul:
                    image = np.asarray(
                        hdul[0].data,
                        dtype=np.float32,
                    )

        except Exception as exc:
            raise ZTFPacketError(f"Could not decode {name}.") from exc

        if image.shape != (63, 63):
            raise ZTFPacketError(f"{name} has unexpected shape {image.shape}.")

        if not np.isfinite(image).all():
            raise ZTFPacketError(f"{name} contains non-finite pixels.")

        return image

    @staticmethod
    def _stamp_bytes(
        cutout: Any,
        *,
        name: str,
    ) -> bytes:
        """Normalize provider-specific cutout representations to bytes."""

        if isinstance(
            cutout,
            Mapping,
        ):
            cutout = cutout.get("stampData")

        if isinstance(
            cutout,
            (
                bytes,
                bytearray,
                memoryview,
            ),
        ):
            return bytes(cutout)

        if isinstance(
            cutout,
            np.ndarray,
        ):
            try:
                return bytes(cutout.tolist())
            except Exception as exc:
                raise ZTFPacketError(f"{name} has invalid byte data.") from exc

        if isinstance(
            cutout,
            Sequence,
        ) and not isinstance(
            cutout,
            (str, bytes),
        ):
            try:
                return bytes(cutout)
            except Exception as exc:
                raise ZTFPacketError(f"{name} has invalid byte data.") from exc

        raise ZTFPacketError(f"{name} does not contain compressed FITS bytes.")


# TODO: Add a separate adapter if future ZTF model generations use
# different alert-index, photometry, metadata, or image conventions.
