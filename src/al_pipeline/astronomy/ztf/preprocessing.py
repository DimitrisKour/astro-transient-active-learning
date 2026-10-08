"""ZTF preprocessing for the historical AstroMiNN input format.

This module reproduces the preprocessing used to construct the 34-feature
metadata vectors and 3-channel image tensors consumed by the historical
AstroMiNN model.

The transformation is intentionally model-version specific. Future ZTF or
Rubin models may use different preprocessing and should receive separate
preprocessor implementations rather than modifying this historical recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


MISSING_VALUE = -999.0

RAW_IMAGE_SHAPE = (63, 63, 3)
MODEL_IMAGE_SHAPE = (3, 63, 63)


# Historical normalization constants used by fixer.py.
METADATA_STATS: dict[str, tuple[float, float]] = {
    "sgscore1": (0.236, 0.266),
    "sgscore2": (0.401, 0.328),
    "distpsnr1": (3.151, 3.757),
    "distpsnr2": (9.269, 6.323),
    "nmtchps": (9.231, 8.089),
    "sharpnr": (0.253, 0.256),
    "scorr": (22.089, 16.757),
    "diffmaglim": (20.17, 0.535),
    "sky": (0.0901, 2.58),
    "ndethist": (18.7, 27.83),
    "ncovhist": (1144.9, 1141.27),
    "sigmapsf": (-20.996, 0.0448),
    "chinr": (3.257, 21.5),
    "magpsf": (18.64, 0.936),
    "nnondet": (1126.0, 1140.0),
    "classtar": (0.95, 0.08),
    "filter_id": (1.62, 0.57),
    "days_since_peak": (10.0, 20.0),
    "days_to_peak": (3.0, 9.0),
    "age": (10.0, 20.0),
    "peakmag_so_far": (17.945, 0.896),
    "maxmag_so_far": (20.37, 0.6325),
}


REQUIRED_METADATA_FIELDS = (
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
    "nnondet",
    "classtar",
    "filter_id",
    "days_since_peak",
    "days_to_peak",
    "age",
    "peakmag_so_far",
    "maxmag_so_far",
    "days_since_peak_fid_1",
    "days_to_peak_fid_1",
    "peakmag_so_far_fid_1",
    "maxmag_so_far_fid_1",
    "days_since_peak_fid_2",
    "days_to_peak_fid_2",
    "peakmag_so_far_fid_2",
    "maxmag_so_far_fid_2",
)


@dataclass(frozen=True)
class PreparedAstroMiNNSample:
    """One ZTF alert prepared for AstroMiNN inference."""

    object_id: str
    alert: int
    image: np.ndarray
    metadata: np.ndarray


class ZTFAstroMiNNPreprocessor:
    """Reproduce the historical ZTF -> AstroMiNN transformation."""

    def preprocess_image(self, images: Any) -> np.ndarray:
        """Convert a 63x63x3 ZTF cutout stack to AstroMiNN channel-first form."""

        array = np.asarray(images)

        if array.shape != RAW_IMAGE_SHAPE:
            raise ValueError(f"Expected raw ZTF image shape {RAW_IMAGE_SHAPE}, got {array.shape}.")

        image = torch.tensor(array, dtype=torch.float32).permute(2, 0, 1).contiguous()

        return image.numpy()

    def preprocess_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        alert_number: int,
    ) -> np.ndarray:
        """Construct the historical 34-feature AstroMiNN metadata vector."""

        missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]

        if missing:
            raise ValueError("Missing required ZTF metadata fields: " + ", ".join(missing))

        def value(name: str) -> float:
            return float(metadata[name])

        def norm(name: str, stats_name: str | None = None) -> float:
            key = stats_name or name
            mean, std = METADATA_STATS[key]
            return (value(name) - mean) / std

        vector = torch.tensor(
            [
                norm("sgscore1"),
                norm("sgscore2"),
                norm("distpsnr1"),
                norm("distpsnr2"),
                norm("nmtchps"),
                norm("sharpnr"),
                norm("scorr"),
                value("ra") / 180.0 - 1.0,
                value("dec") / 90.0,
                norm("diffmaglim"),
                norm("sky"),
                norm("ndethist"),
                norm("ncovhist"),
                norm("sigmapsf"),
                norm("chinr"),
                norm("magpsf"),
                norm("nnondet"),
                norm("classtar"),
                norm("filter_id"),
                norm("days_since_peak"),
                norm("days_to_peak"),
                norm("age"),
                norm("peakmag_so_far"),
                norm("maxmag_so_far"),
                value("maxmag_so_far") / value("peakmag_so_far"),
                norm(
                    "days_since_peak_fid_1",
                    "days_since_peak",
                ),
                norm(
                    "days_to_peak_fid_1",
                    "days_to_peak",
                ),
                norm(
                    "peakmag_so_far_fid_1",
                    "peakmag_so_far",
                ),
                norm(
                    "maxmag_so_far_fid_1",
                    "maxmag_so_far",
                ),
                norm(
                    "days_since_peak_fid_2",
                    "days_since_peak",
                ),
                norm(
                    "days_to_peak_fid_2",
                    "days_to_peak",
                ),
                norm(
                    "peakmag_so_far_fid_2",
                    "peakmag_so_far",
                ),
                norm(
                    "maxmag_so_far_fid_2",
                    "maxmag_so_far",
                ),
                float(alert_number),
            ],
            dtype=torch.float32,
        )

        vector = torch.nan_to_num(
            vector,
            nan=MISSING_VALUE,
        )

        return vector.numpy()

    def preprocess(
        self,
        *,
        object_id: str,
        alert_number: int,
        images: Any,
        metadata: Mapping[str, Any],
    ) -> PreparedAstroMiNNSample:
        """Prepare one canonical ZTF alert for AstroMiNN."""

        return PreparedAstroMiNNSample(
            object_id=str(object_id),
            alert=int(alert_number),
            image=self.preprocess_image(images),
            metadata=self.preprocess_metadata(
                metadata,
                alert_number=alert_number,
            ),
        )

    def preprocess_legacy_sample(
        self,
        sample: Mapping[str, Any],
    ) -> PreparedAstroMiNNSample:
        """Prepare one historical raw-files sample.

        Labels are deliberately not propagated. Pool preprocessing must not
        expose hidden labels to the active-learning pipeline.
        """

        return self.preprocess(
            object_id=str(sample["obj_id"]),
            alert_number=int(sample["alerte"]),
            images=sample["images"],
            metadata=sample["metadata"],
        )


# TODO: Add additional model-version-specific preprocessors rather than
# changing this historical recipe when future AstroMiNN inputs differ.
