"""Photometry-derived ZTF features for historical AstroMiNN inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import math
import numpy as np


# The historical 34-feature AstroMiNN representation only consumes
# per-filter derived quantities for ZTF fid 1 and fid 2.
ASTROMINN_FIDS = (1, 2)


class ZTFPhotometryFeatureBuilder:
    """Build the light-curve features expected by historical AstroMiNN.

    The supplied photometry history must contain all usable detections up to
    and including the alert currently being prepared.

    Each photometry row should provide:
        jd
        magpsf
        fid

    The timing definitions intentionally reproduce the historical pipeline,
    including its use of global first/latest observation times for the
    per-filter timing features.
    """

    def build(
        self,
        photometry: Iterable[Mapping[str, Any]],
    ) -> dict[str, float]:
        """Return AstroMiNN photometry-derived metadata."""

        rows = self._clean_rows(photometry)

        if not rows:
            return self._empty_features()

        first_jd = min(row["jd"] for row in rows)
        last_jd = max(row["jd"] for row in rows)

        # Historical global behavior:
        # idxmin() on the unsorted DataFrame selected the first occurrence
        # of the brightest magnitude.
        peak_row = min(
            rows,
            key=lambda row: row["magpsf"],
        )

        result = {
            "days_since_peak": float(last_jd - peak_row["jd"]),
            "days_to_peak": float(peak_row["jd"] - first_jd),
            "age": float(last_jd - first_jd),
            "peakmag_so_far": float(peak_row["magpsf"]),
            "maxmag_so_far": float(max(row["magpsf"] for row in rows)),
        }

        for fid in ASTROMINN_FIDS:
            filter_rows = [row for row in rows if row["fid"] == fid]

            if not filter_rows:
                result.update(self._empty_filter_features(fid))
                continue

            # Historical code sorted each filter by time before idxmin().
            # Therefore an equal-magnitude tie resolves to the earliest JD.
            filter_rows = sorted(
                filter_rows,
                key=lambda row: row["jd"],
            )

            filter_peak = min(
                filter_rows,
                key=lambda row: row["magpsf"],
            )

            result.update(
                {
                    f"days_since_peak_fid_{fid}": float(last_jd - filter_peak["jd"]),
                    f"days_to_peak_fid_{fid}": float(filter_peak["jd"] - first_jd),
                    f"peakmag_so_far_fid_{fid}": float(filter_peak["magpsf"]),
                    f"maxmag_so_far_fid_{fid}": float(max(row["magpsf"] for row in filter_rows)),
                }
            )

        return result

    def augment_metadata(
        self,
        metadata: Mapping[str, Any],
        photometry: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return candidate metadata augmented with derived photometry fields."""

        result = dict(metadata)
        result.update(self.build(photometry))
        return result

    @staticmethod
    def _clean_rows(
        photometry: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, float | int | None]]:
        """Normalize usable photometry rows.

        Historical preprocessing discarded rows with unusable jd/magpsf
        values. Missing or invalid fid values remain usable for global
        metrics but cannot contribute to per-filter metrics.
        """

        rows: list[dict[str, float | int | None]] = []

        for point in photometry:
            if "jd" not in point:
                raise ValueError("Photometry row is missing 'jd'.")

            if "magpsf" not in point:
                raise ValueError("Photometry row is missing 'magpsf'.")

            try:
                jd = float(point["jd"])
                magpsf = float(point["magpsf"])
            except (TypeError, ValueError):
                continue

            if math.isnan(jd) or math.isnan(magpsf):
                continue

            fid: int | None

            try:
                raw_fid = point.get("fid")

                if raw_fid is None:
                    fid = None
                else:
                    numeric_fid = float(raw_fid)

                    if math.isnan(numeric_fid):
                        fid = None
                    else:
                        fid = int(numeric_fid)

            except (TypeError, ValueError):
                fid = None

            rows.append(
                {
                    "jd": jd,
                    "magpsf": magpsf,
                    "fid": fid,
                }
            )

        return rows

    @staticmethod
    def _empty_filter_features(
        fid: int,
    ) -> dict[str, float]:
        return {
            f"days_since_peak_fid_{fid}": np.nan,
            f"days_to_peak_fid_{fid}": np.nan,
            f"peakmag_so_far_fid_{fid}": np.nan,
            f"maxmag_so_far_fid_{fid}": np.nan,
        }

    def _empty_features(
        self,
    ) -> dict[str, float]:
        result = {
            "days_since_peak": np.nan,
            "days_to_peak": np.nan,
            "age": np.nan,
            "peakmag_so_far": np.nan,
            "maxmag_so_far": np.nan,
        }

        for fid in ASTROMINN_FIDS:
            result.update(self._empty_filter_features(fid))

        return result


# TODO: Rubin and future ZTF model generations may require different
# photometric summaries. Add separate builders rather than changing the
# historical AstroMiNN definitions encoded here.
