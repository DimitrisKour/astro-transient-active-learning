import numpy as np

from al_pipeline.astronomy.ztf.photometry import ZTFPhotometryFeatureBuilder


def test_global_and_per_filter_features():
    builder = ZTFPhotometryFeatureBuilder()

    photometry = [
        {
            "jd": 100.0,
            "magpsf": 20.0,
            "fid": 1,
        },
        {
            "jd": 102.0,
            "magpsf": 19.0,
            "fid": 2,
        },
        {
            "jd": 105.0,
            "magpsf": 18.0,
            "fid": 1,
        },
        {
            "jd": 107.0,
            "magpsf": 20.5,
            "fid": 2,
        },
    ]

    result = builder.build(photometry)

    assert result["days_since_peak"] == 2.0
    assert result["days_to_peak"] == 5.0
    assert result["age"] == 7.0
    assert result["peakmag_so_far"] == 18.0
    assert result["maxmag_so_far"] == 20.5

    assert result["days_since_peak_fid_1"] == 2.0
    assert result["days_to_peak_fid_1"] == 5.0
    assert result["peakmag_so_far_fid_1"] == 18.0
    assert result["maxmag_so_far_fid_1"] == 20.0

    # Historical behavior uses the global first/latest JD here.
    assert result["days_since_peak_fid_2"] == 5.0
    assert result["days_to_peak_fid_2"] == 2.0
    assert result["peakmag_so_far_fid_2"] == 19.0
    assert result["maxmag_so_far_fid_2"] == 20.5


def test_single_detection():
    builder = ZTFPhotometryFeatureBuilder()

    result = builder.build(
        [
            {
                "jd": 100.0,
                "magpsf": 19.5,
                "fid": 1,
            }
        ]
    )

    assert result["days_since_peak"] == 0.0
    assert result["days_to_peak"] == 0.0
    assert result["age"] == 0.0
    assert result["peakmag_so_far"] == 19.5
    assert result["maxmag_so_far"] == 19.5

    assert result["days_since_peak_fid_1"] == 0.0
    assert result["days_to_peak_fid_1"] == 0.0

    assert np.isnan(result["days_since_peak_fid_2"])


def test_missing_filter_produces_nan_features():
    builder = ZTFPhotometryFeatureBuilder()

    result = builder.build(
        [
            {
                "jd": 100.0,
                "magpsf": 19.5,
                "fid": 1,
            },
            {
                "jd": 101.0,
                "magpsf": 19.0,
                "fid": 1,
            },
        ]
    )

    assert np.isnan(result["days_since_peak_fid_2"])
    assert np.isnan(result["days_to_peak_fid_2"])
    assert np.isnan(result["peakmag_so_far_fid_2"])
    assert np.isnan(result["maxmag_so_far_fid_2"])


def test_invalid_photometry_is_discarded():
    builder = ZTFPhotometryFeatureBuilder()

    result = builder.build(
        [
            {
                "jd": np.nan,
                "magpsf": 17.0,
                "fid": 1,
            },
            {
                "jd": 100.0,
                "magpsf": np.nan,
                "fid": 1,
            },
            {
                "jd": 101.0,
                "magpsf": 19.0,
                "fid": 1,
            },
        ]
    )

    assert result["age"] == 0.0
    assert result["peakmag_so_far"] == 19.0


def test_empty_history_returns_nan_features():
    builder = ZTFPhotometryFeatureBuilder()

    result = builder.build([])

    assert np.isnan(result["days_since_peak"])
    assert np.isnan(result["age"])
    assert np.isnan(result["peakmag_so_far_fid_1"])


def test_augment_metadata_preserves_candidate_fields():
    builder = ZTFPhotometryFeatureBuilder()

    metadata = {
        "ra": 123.0,
        "dec": -4.0,
    }

    result = builder.augment_metadata(
        metadata,
        [
            {
                "jd": 100.0,
                "magpsf": 19.0,
                "fid": 1,
            }
        ],
    )

    assert result["ra"] == 123.0
    assert result["dec"] == -4.0
    assert result["age"] == 0.0
    assert result["peakmag_so_far"] == 19.0
