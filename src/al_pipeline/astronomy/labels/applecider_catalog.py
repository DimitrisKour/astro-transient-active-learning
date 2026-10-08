"""Conservative AppleCiDEr rules for BOOM catalogue evidence."""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from al_pipeline.core.evidence.types import Evidence
from al_pipeline.core.oracle.resolution import LabelStatus
from al_pipeline.core.oracle.rule_based import RuleBasedLabelResolver
from al_pipeline.core.oracle.rules import ResolutionClaim


TNS_SN_I_TYPES = frozenset({"SN IA", "SN IA-PEC", "SN IB", "SN IC", "SN IC-BL"})
TNS_SN_II_TYPES = frozenset({"SN II", "SN IIP", "SN IIN"})
TNS_CV_TYPES = frozenset({"CV"})
TNS_TDE_TYPES = frozenset({"TDE"})

VSX_CV_TYPES = frozenset({"UG", "UG:", "UGSU", "UGSU:", "UGWZ", "UGWZ:"})
VSX_SN_I_TYPES = frozenset({"SN IB", "SN IC"})

MILLIQUAS_AGN_TYPES = frozenset({"A", "AR", "ARX", "AX", "Q", "QR", "QX"})


def _catalog_row(evidence: Evidence) -> Mapping[str, Any] | None:
    if evidence.kind != "catalog_match":
        return None

    if not isinstance(evidence.value, Mapping):
        return None

    return evidence.value


def _normalized_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _normalized_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        return ()

    return tuple(
        dict.fromkeys(normalized for item in values if (normalized := _normalized_text(item)))
    )


def _separation_within(
    evidence: Evidence,
    max_arcsec: float,
) -> bool:
    value = evidence.provenance.get("separation_arcsec")

    try:
        separation = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(separation) and 0.0 <= separation <= max_arcsec


def _native_label(values: Sequence[str]) -> object:
    return values[0] if len(values) == 1 else tuple(values)


@dataclass(frozen=True, slots=True)
class TNSLabelRule:
    """Map exact-name TNS classifications into the AppleCiDEr taxonomy.

    BOOM performs a positional cone search, which can return unrelated nearby
    TNS objects. A TNS row is therefore resolving only when its
    ``internal_names`` explicitly contains the selected ZTF object ID.
    """

    sn_i_label: object = "SN I"
    sn_ii_label: object = "SN II"
    cv_label: object = "CV"
    tde_label: object = "TDE"
    max_separation_arcsec: float = 5.0

    def __post_init__(self) -> None:
        if self.max_separation_arcsec <= 0:
            raise ValueError("max_separation_arcsec must be > 0.")

    def evaluate(self, evidence: Evidence) -> ResolutionClaim | None:
        if evidence.source != "TNS":
            return None

        row = _catalog_row(evidence)
        if row is None:
            return None

        if not _separation_within(evidence, self.max_separation_arcsec):
            return None

        internal_names = _normalized_values(row.get("internal_names"))
        object_id = _normalized_text(evidence.object_id)

        if object_id not in internal_names:
            return None

        native = _normalized_text(row.get("type"))

        if native in TNS_SN_I_TYPES:
            target = self.sn_i_label
        elif native in TNS_SN_II_TYPES:
            target = self.sn_ii_label
        elif native in TNS_CV_TYPES:
            target = self.cv_label
        elif native in TNS_TDE_TYPES:
            target = self.tde_label
        else:
            # This intentionally leaves blank types and SLSN-II unresolved.
            return None

        return ResolutionClaim(
            object_id=evidence.object_id,
            status=LabelStatus.LABELED,
            label=target,
            source=evidence.source,
            native_label=native,
            rule=type(self).__name__,
            provenance={
                "separation_arcsec": evidence.provenance.get("separation_arcsec"),
                "catalog_id": row.get("_id"),
                "tns_name": row.get("name"),
                "tns_name_prefix": row.get("name_prefix"),
                "exact_internal_name_match": True,
            },
        )


@dataclass(frozen=True, slots=True)
class VSXTargetRule:
    """Map close, unambiguous VSX CV and type-I supernova classes."""

    sn_i_label: object = "SN I"
    cv_label: object = "CV"
    max_separation_arcsec: float = 1.0

    def __post_init__(self) -> None:
        if self.max_separation_arcsec <= 0:
            raise ValueError("max_separation_arcsec must be > 0.")

    def evaluate(self, evidence: Evidence) -> ResolutionClaim | None:
        if evidence.source != "VSX":
            return None

        row = _catalog_row(evidence)
        if row is None:
            return None

        if not _separation_within(evidence, self.max_separation_arcsec):
            return None

        native_types = _normalized_values(row.get("types"))
        if not native_types:
            return None

        targets: list[object] = []
        for native in native_types:
            if native in VSX_CV_TYPES:
                targets.append(self.cv_label)
            elif native in VSX_SN_I_TYPES:
                targets.append(self.sn_i_label)
            else:
                # Mixed values such as ["SN", "UG"] are not decisive.
                return None

        first_target = targets[0]
        if any(target != first_target for target in targets[1:]):
            return None

        return ResolutionClaim(
            object_id=evidence.object_id,
            status=LabelStatus.LABELED,
            label=first_target,
            source=evidence.source,
            native_label=_native_label(native_types),
            rule=type(self).__name__,
            provenance={
                "separation_arcsec": evidence.provenance.get("separation_arcsec"),
                "catalog_id": row.get("_id"),
                "name": row.get("name"),
            },
        )


@dataclass(frozen=True, slots=True)
class MilliquasAGNRule:
    """Map close, explicitly audited Milliquas identities to AGN."""

    agn_label: object = "AGN"
    max_separation_arcsec: float = 2.0
    agn_types: frozenset[str] = MILLIQUAS_AGN_TYPES

    def __post_init__(self) -> None:
        if self.max_separation_arcsec <= 0:
            raise ValueError("max_separation_arcsec must be > 0.")

        normalized = frozenset(
            _normalized_text(value) for value in self.agn_types if _normalized_text(value)
        )
        if not normalized:
            raise ValueError("agn_types must not be empty.")
        object.__setattr__(self, "agn_types", normalized)

    def evaluate(self, evidence: Evidence) -> ResolutionClaim | None:
        if evidence.source != "milliquas_v8":
            return None

        row = _catalog_row(evidence)
        if row is None:
            return None

        if not _separation_within(evidence, self.max_separation_arcsec):
            return None

        native = _normalized_text(row.get("objtype"))

        if native not in self.agn_types:
            return None

        return ResolutionClaim(
            object_id=evidence.object_id,
            status=LabelStatus.LABELED,
            label=self.agn_label,
            source=evidence.source,
            native_label=native,
            rule=type(self).__name__,
            provenance={
                "separation_arcsec": evidence.provenance.get("separation_arcsec"),
                "catalog_id": row.get("_id"),
                "redshift": row.get("z"),
            },
        )


@dataclass(frozen=True, slots=True)
class DESIQSORule:
    """Map close, warning-free DESI QSO spectra to AGN."""

    agn_label: object = "AGN"
    max_separation_arcsec: float = 2.0

    def __post_init__(self) -> None:
        if self.max_separation_arcsec <= 0:
            raise ValueError("max_separation_arcsec must be > 0.")

    def evaluate(self, evidence: Evidence) -> ResolutionClaim | None:
        if evidence.source != "DESI_DR1":
            return None

        row = _catalog_row(evidence)
        if row is None:
            return None

        if not _separation_within(evidence, self.max_separation_arcsec):
            return None

        if _normalized_text(row.get("spectype")) != "QSO":
            return None

        try:
            zwarn = int(row.get("zwarn"))
        except (TypeError, ValueError):
            return None

        if zwarn != 0:
            return None

        return ResolutionClaim(
            object_id=evidence.object_id,
            status=LabelStatus.LABELED,
            label=self.agn_label,
            source=evidence.source,
            native_label="QSO",
            rule=type(self).__name__,
            provenance={
                "separation_arcsec": evidence.provenance.get("separation_arcsec"),
                "catalog_id": row.get("_id"),
                "redshift": row.get("z"),
                "zwarn": zwarn,
            },
        )


@dataclass(frozen=True, slots=True)
class NEDQSORule:
    """Map close NED QSO identities to AGN, but not generic hosts."""

    agn_label: object = "AGN"
    max_separation_arcsec: float = 1.0

    def __post_init__(self) -> None:
        if self.max_separation_arcsec <= 0:
            raise ValueError("max_separation_arcsec must be > 0.")

    def evaluate(self, evidence: Evidence) -> ResolutionClaim | None:
        if evidence.source != "NED":
            return None

        row = _catalog_row(evidence)
        if row is None:
            return None

        if not _separation_within(evidence, self.max_separation_arcsec):
            return None

        if _normalized_text(row.get("objtype")) != "QSO":
            return None

        return ResolutionClaim(
            object_id=evidence.object_id,
            status=LabelStatus.LABELED,
            label=self.agn_label,
            source=evidence.source,
            native_label="QSO",
            rule=type(self).__name__,
            provenance={
                "separation_arcsec": evidence.provenance.get("separation_arcsec"),
                "catalog_id": row.get("_id"),
                "redshift": row.get("z"),
            },
        )


@dataclass(frozen=True, slots=True)
class VSXOutsideRule:
    """Resolve close, explicitly configured VSX classes as outside taxonomy."""

    outside_types: frozenset[str] = frozenset({"RRAB", "ROT", "RS"})
    max_separation_arcsec: float = 1.0

    def __post_init__(self) -> None:
        if self.max_separation_arcsec <= 0:
            raise ValueError("max_separation_arcsec must be > 0.")

        normalized = frozenset(
            _normalized_text(value) for value in self.outside_types if _normalized_text(value)
        )

        if not normalized:
            raise ValueError("outside_types must not be empty.")

        object.__setattr__(self, "outside_types", normalized)

    def evaluate(self, evidence: Evidence) -> ResolutionClaim | None:
        if evidence.source != "VSX":
            return None

        row = _catalog_row(evidence)
        if row is None:
            return None

        if not _separation_within(evidence, self.max_separation_arcsec):
            return None

        native_types = _normalized_values(row.get("types"))

        # A mixed supported/unsupported type list is not decisive evidence.
        if not native_types or any(native not in self.outside_types for native in native_types):
            return None

        return ResolutionClaim(
            object_id=evidence.object_id,
            status=LabelStatus.OUTSIDE_LABEL_SPACE,
            source=evidence.source,
            native_label=_native_label(native_types),
            rule=type(self).__name__,
            provenance={
                "separation_arcsec": evidence.provenance.get("separation_arcsec"),
                "catalog_id": row.get("_id"),
                "name": row.get("name"),
            },
        )


class AppleCiDErCatalogResolver(RuleBasedLabelResolver):
    """Evidence-backed resolver for the BOOM catalogues used by AppleCiDEr.

    Only mappings validated by the historical BOOM calibration are enabled.
    Generic host classes and unsupported or ambiguous native classes remain
    unresolved rather than being coerced into the target taxonomy.
    """

    def __init__(
        self,
        *,
        agn_label: object = "AGN",
        sn_i_label: object = "SN I",
        sn_ii_label: object = "SN II",
        cv_label: object = "CV",
        tde_label: object = "TDE",
        tns_max_separation_arcsec: float = 5.0,
        milliquas_max_separation_arcsec: float = 2.0,
        milliquas_agn_types: frozenset[str] = MILLIQUAS_AGN_TYPES,
        ned_max_separation_arcsec: float = 1.0,
        desi_max_separation_arcsec: float = 2.0,
        vsx_max_separation_arcsec: float = 1.0,
        vsx_outside_types: frozenset[str] = frozenset({"RRAB", "ROT", "RS"}),
    ) -> None:
        super().__init__(
            rules=(
                TNSLabelRule(
                    sn_i_label=sn_i_label,
                    sn_ii_label=sn_ii_label,
                    cv_label=cv_label,
                    tde_label=tde_label,
                    max_separation_arcsec=tns_max_separation_arcsec,
                ),
                VSXTargetRule(
                    sn_i_label=sn_i_label,
                    cv_label=cv_label,
                    max_separation_arcsec=vsx_max_separation_arcsec,
                ),
                MilliquasAGNRule(
                    agn_label=agn_label,
                    max_separation_arcsec=(milliquas_max_separation_arcsec),
                    agn_types=milliquas_agn_types,
                ),
                NEDQSORule(
                    agn_label=agn_label,
                    max_separation_arcsec=ned_max_separation_arcsec,
                ),
                DESIQSORule(
                    agn_label=agn_label,
                    max_separation_arcsec=desi_max_separation_arcsec,
                ),
                VSXOutsideRule(
                    outside_types=vsx_outside_types,
                    max_separation_arcsec=vsx_max_separation_arcsec,
                ),
            )
        )
