"""Tests for the current evidence-backed AppleCiDEr catalogue policy."""

from al_pipeline.core.evidence.types import Evidence, EvidenceBatch
from al_pipeline.astronomy.labels.applecider_catalog import AppleCiDErCatalogResolver
from al_pipeline.core.oracle.resolution import LabelStatus


def _catalog_evidence(object_id, source, row, separation):
    return Evidence(
        object_id=object_id,
        source=source,
        kind="catalog_match",
        value=row,
        provenance={"separation_arcsec": separation},
    )


def test_tns_exact_identity_maps_audited_types_to_target_taxonomy():
    resolver = AppleCiDErCatalogResolver()

    native_to_target = {
        "CV": "CV",
        "SN II": "SN II",
        "SN IIP": "SN II",
        "SN IIn": "SN II",
        "SN Ia": "SN I",
        "SN Ia-pec": "SN I",
        "SN Ib": "SN I",
        "SN Ic": "SN I",
        "SN Ic-BL": "SN I",
        "TDE": "TDE",
    }

    batch = EvidenceBatch(
        requested_ids=list(native_to_target),
        evidence=[
            _catalog_evidence(
                native,
                "TNS",
                {
                    "_id": native,
                    "internal_names": [native],
                    "name": "2026abc",
                    "type": native,
                },
                0.2,
            )
            for native in native_to_target
        ],
    )

    results = {result.object_id: result for result in resolver.resolve(batch).resolutions}

    assert {native: results[native].label for native in native_to_target} == native_to_target
    assert all(result.status is LabelStatus.LABELED for result in results.values())


def test_tns_ignores_nearby_row_without_selected_ztf_internal_name():
    resolver = AppleCiDErCatalogResolver()

    result = resolver.resolve(
        EvidenceBatch(
            requested_ids=["ZTF-selected"],
            evidence=[
                _catalog_evidence(
                    "ZTF-selected",
                    "TNS",
                    {
                        "internal_names": ["ZTF-neighbour"],
                        "type": "SN Ia",
                    },
                    0.05,
                ),
                _catalog_evidence(
                    "ZTF-selected",
                    "TNS",
                    {
                        "internal_names": ["ZTF-selected"],
                        "type": "SN II",
                    },
                    0.2,
                ),
            ],
        )
    ).resolutions[0]

    assert result.status is LabelStatus.LABELED
    assert result.label == "SN II"
    assert result.provenance["claims"][0]["provenance"]["exact_internal_name_match"] is True


def test_tns_unsupported_or_untyped_rows_remain_unresolved():
    resolver = AppleCiDErCatalogResolver()

    batch = EvidenceBatch(
        requested_ids=["slsn", "untyped", "no-id"],
        evidence=[
            _catalog_evidence(
                "slsn",
                "TNS",
                {"internal_names": ["slsn"], "type": "SLSN-II"},
                0.1,
            ),
            _catalog_evidence(
                "untyped",
                "TNS",
                {"internal_names": ["untyped"], "type": None},
                0.1,
            ),
            _catalog_evidence(
                "no-id",
                "TNS",
                {"type": "SN Ia"},
                0.1,
            ),
        ],
    )

    assert all(
        result.status is LabelStatus.UNRESOLVED for result in resolver.resolve(batch).resolutions
    )


def test_milliquas_audited_agn_types_resolve_to_agn():
    resolver = AppleCiDErCatalogResolver()
    supported = ("A", "AR", "ARX", "AX", "Q", "QR", "QX")

    batch = EvidenceBatch(
        requested_ids=list(supported),
        evidence=[
            _catalog_evidence(
                native,
                "milliquas_v8",
                {"_id": native, "objtype": native, "z": 0.32},
                1.40,
            )
            for native in supported
        ],
    )

    results = resolver.resolve(batch)

    assert all(result.status is LabelStatus.LABELED for result in results.resolutions)
    assert all(result.label == "AGN" for result in results.resolutions)


def test_milliquas_unsupported_n_and_nx_types_remain_unresolved():
    resolver = AppleCiDErCatalogResolver()

    batch = EvidenceBatch(
        requested_ids=["n", "nx"],
        evidence=[
            _catalog_evidence(
                "n",
                "milliquas_v8",
                {"objtype": "N"},
                0.4,
            ),
            _catalog_evidence(
                "nx",
                "milliquas_v8",
                {"objtype": "NX"},
                0.4,
            ),
        ],
    )

    assert all(
        result.status is LabelStatus.UNRESOLVED for result in resolver.resolve(batch).resolutions
    )


def test_desi_qso_requires_clean_spectrum_and_close_association():
    resolver = AppleCiDErCatalogResolver()

    batch = EvidenceBatch(
        requested_ids=["good", "warn", "star", "far"],
        evidence=[
            _catalog_evidence(
                "good",
                "DESI_DR1",
                {"spectype": "QSO", "zwarn": 0, "z": 1.718},
                0.49,
            ),
            _catalog_evidence(
                "warn",
                "DESI_DR1",
                {"spectype": "QSO", "zwarn": 4},
                0.49,
            ),
            _catalog_evidence(
                "star",
                "DESI_DR1",
                {"spectype": "STAR", "zwarn": 0},
                0.2,
            ),
            _catalog_evidence(
                "far",
                "DESI_DR1",
                {"spectype": "QSO", "zwarn": 0},
                3.0,
            ),
        ],
    )

    results = {result.object_id: result for result in resolver.resolve(batch).resolutions}

    assert results["good"].status is LabelStatus.LABELED
    assert results["good"].label == "AGN"
    assert results["warn"].status is LabelStatus.UNRESOLVED
    assert results["star"].status is LabelStatus.UNRESOLVED
    assert results["far"].status is LabelStatus.UNRESOLVED


def test_ned_qso_resolves_to_agn_but_generic_host_does_not():
    resolver = AppleCiDErCatalogResolver()

    batch = EvidenceBatch(
        requested_ids=["qso", "host", "far"],
        evidence=[
            _catalog_evidence(
                "qso",
                "NED",
                {"_id": "Q-1", "objtype": "QSO", "z": 0.8},
                0.4,
            ),
            _catalog_evidence(
                "host",
                "NED",
                {"_id": "G-1", "objtype": "G", "z": 0.03},
                0.2,
            ),
            _catalog_evidence(
                "far",
                "NED",
                {"_id": "Q-2", "objtype": "QSO", "z": 0.9},
                1.2,
            ),
        ],
    )

    results = {result.object_id: result for result in resolver.resolve(batch).resolutions}

    assert results["qso"].status is LabelStatus.LABELED
    assert results["qso"].label == "AGN"
    assert results["host"].status is LabelStatus.UNRESOLVED
    assert results["far"].status is LabelStatus.UNRESOLVED


def test_close_vsx_audited_target_types_resolve_to_target_taxonomy():
    resolver = AppleCiDErCatalogResolver()
    native_to_target = {
        "UG": "CV",
        "UG:": "CV",
        "UGSU": "CV",
        "UGSU:": "CV",
        "UGWZ": "CV",
        "UGWZ:": "CV",
        "SN Ib": "SN I",
        "SN Ic": "SN I",
    }

    batch = EvidenceBatch(
        requested_ids=list(native_to_target),
        evidence=[
            _catalog_evidence(
                native,
                "VSX",
                {"types": [native]},
                0.2,
            )
            for native in native_to_target
        ],
    )

    results = {result.object_id: result for result in resolver.resolve(batch).resolutions}

    assert {native: results[native].label for native in native_to_target} == native_to_target


def test_vsx_mixed_sn_and_ug_type_remains_unresolved():
    resolver = AppleCiDErCatalogResolver()

    result = resolver.resolve(
        EvidenceBatch(
            requested_ids=["obj"],
            evidence=[
                _catalog_evidence(
                    "obj",
                    "VSX",
                    {"types": ["SN", "UG"]},
                    0.1,
                )
            ],
        )
    ).resolutions[0]

    assert result.status is LabelStatus.UNRESOLVED


def test_close_vsx_audited_types_resolve_outside_current_taxonomy():
    resolver = AppleCiDErCatalogResolver()

    batch = EvidenceBatch(
        requested_ids=["rr", "rot", "rs"],
        evidence=[
            _catalog_evidence("rr", "VSX", {"types": ["RRAB"]}, 0.25),
            _catalog_evidence("rot", "VSX", {"types": ["ROT"]}, 0.53),
            _catalog_evidence("rs", "VSX", {"types": ["RS"]}, 0.06),
        ],
    )

    results = resolver.resolve(batch).resolutions

    assert all(result.status is LabelStatus.OUTSIDE_LABEL_SPACE for result in results)


def test_vsx_rule_is_conservative_for_broader_or_unseen_matches():
    resolver = AppleCiDErCatalogResolver()

    batch = EvidenceBatch(
        requested_ids=["far", "unknown"],
        evidence=[
            _catalog_evidence("far", "VSX", {"types": ["ROT"]}, 1.55),
            _catalog_evidence("unknown", "VSX", {"types": ["EA"]}, 0.2),
        ],
    )

    results = resolver.resolve(batch).resolutions

    assert all(result.status is LabelStatus.UNRESOLVED for result in results)


def test_vsx_mixed_outside_and_unsupported_type_remains_unresolved():
    resolver = AppleCiDErCatalogResolver()

    result = resolver.resolve(
        EvidenceBatch(
            requested_ids=["obj"],
            evidence=[
                _catalog_evidence(
                    "obj",
                    "VSX",
                    {"types": ["RRAB", "SN"]},
                    0.2,
                )
            ],
        )
    ).resolutions[0]

    assert result.status is LabelStatus.UNRESOLVED


def test_context_only_ned_match_does_not_resolve():
    resolver = AppleCiDErCatalogResolver()

    result = resolver.resolve(
        EvidenceBatch(
            requested_ids=["obj"],
            evidence=[
                _catalog_evidence(
                    "obj",
                    "NED",
                    {"objtype": "G", "z": 0.03},
                    0.9,
                )
            ],
        )
    ).resolutions[0]

    assert result.status is LabelStatus.UNRESOLVED


def test_agreeing_milliquas_and_desi_evidence_combines_cleanly():
    resolver = AppleCiDErCatalogResolver()

    result = resolver.resolve(
        EvidenceBatch(
            requested_ids=["obj"],
            evidence=[
                _catalog_evidence(
                    "obj",
                    "milliquas_v8",
                    {"objtype": "Q", "z": 1.716},
                    0.49,
                ),
                _catalog_evidence(
                    "obj",
                    "DESI_DR1",
                    {"spectype": "QSO", "zwarn": 0, "z": 1.718},
                    0.49,
                ),
            ],
        )
    ).resolutions[0]

    assert result.status is LabelStatus.LABELED
    assert result.label == "AGN"
    assert result.source == "multiple"
    assert result.native_label == ("Q", "QSO")


def test_conflicting_target_and_outside_evidence_stays_unresolved():
    resolver = AppleCiDErCatalogResolver()

    result = resolver.resolve(
        EvidenceBatch(
            requested_ids=["obj"],
            evidence=[
                _catalog_evidence(
                    "obj",
                    "milliquas_v8",
                    {"objtype": "Q"},
                    0.4,
                ),
                _catalog_evidence(
                    "obj",
                    "VSX",
                    {"types": ["RRAB"]},
                    0.3,
                ),
            ],
        )
    ).resolutions[0]

    assert result.status is LabelStatus.UNRESOLVED
    assert result.provenance["reason"] == "conflicting_resolving_evidence"
