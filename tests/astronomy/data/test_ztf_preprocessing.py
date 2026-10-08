import numpy as np
import pytest

from al_pipeline.astronomy.ztf.preprocessing import ZTFAstroMiNNPreprocessor


def historical_metadata():
    return {
        "sgscore1": 0.9991670250892639,
        "sgscore2": 0.01754169911146164,
        "distpsnr1": 0.27526527643203735,
        "distpsnr2": 17.970918655395508,
        "ra": 3.909512,
        "dec": 26.6157178,
        "nmtchps": 7.0,
        "sharpnr": 0.006000000052154064,
        "scorr": 12.5969638824463,
        "sky": 0.06701137870550156,
        "diffmaglim": 20.8050537109375,
        "ndethist": 4.0,
        "ncovhist": 11.0,
        "sigmapsf": 0.19124293327331543,
        "chinr": 0.8759999871253967,
        "magpsf": 19.841123580932617,
        "nnondet": 7.0,
        "classtar": 0.9879999756813049,
        "filter_id": 1.0,
        "days_since_peak": 0.0,
        "days_to_peak": 6.088738400023431,
        "age": 6.088738400023431,
        "peakmag_so_far": 19.841123580932617,
        "maxmag_so_far": 20.687299728393555,
        "days_since_peak_fid_1": 0.0,
        "days_to_peak_fid_1": 6.088738400023431,
        "peakmag_so_far_fid_1": 19.841123580932617,
        "maxmag_so_far_fid_1": 20.687299728393555,
        "days_since_peak_fid_2": np.nan,
        "days_to_peak_fid_2": np.nan,
        "peakmag_so_far_fid_2": np.nan,
        "maxmag_so_far_fid_2": np.nan,
    }


def test_historical_metadata_regression():
    preprocessor = ZTFAstroMiNNPreprocessor()

    result = preprocessor.preprocess_metadata(
        historical_metadata(),
        alert_number=0,
    )

    expected = np.array(
        [
            2.869049072265625,
            -1.169080138206482,
            -0.7654337882995605,
            1.376232624053955,
            -0.2758066654205322,
            -0.96484375,
            -0.5664520263671875,
            -0.9782804846763611,
            0.29573020339012146,
            1.187016248703003,
            -0.008949077688157558,
            -0.5282069444656372,
            -0.9935422539710999,
            472.9295349121094,
            -0.1107441857457161,
            1.2832516431808472,
            -0.9815789461135864,
            0.47499969601631165,
            -1.0877193212509155,
            -0.5,
            0.3431931436061859,
            -0.19556307792663574,
            2.1162092685699463,
            0.501659631729126,
            1.0426476001739502,
            -0.5,
            0.3431931436061859,
            2.1162092685699463,
            0.501659631729126,
            -999.0,
            -999.0,
            -999.0,
            -999.0,
            0.0,
        ],
        dtype=np.float32,
    )

    assert result.shape == (34,)
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)


def test_image_channel_order():
    preprocessor = ZTFAstroMiNNPreprocessor()

    raw = np.arange(
        63 * 63 * 3,
        dtype=np.float32,
    ).reshape(63, 63, 3)

    result = preprocessor.preprocess_image(raw)

    expected = np.transpose(
        raw,
        (2, 0, 1),
    )

    assert result.shape == (3, 63, 63)
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)


def test_rejects_wrong_image_shape():
    preprocessor = ZTFAstroMiNNPreprocessor()

    with pytest.raises(
        ValueError,
        match="Expected raw ZTF image shape",
    ):
        preprocessor.preprocess_image(
            np.zeros(
                (45, 45, 3),
                dtype=np.float32,
            )
        )


def test_missing_metadata_field_is_rejected():
    preprocessor = ZTFAstroMiNNPreprocessor()

    metadata = historical_metadata()
    del metadata["sgscore1"]

    with pytest.raises(
        ValueError,
        match="sgscore1",
    ):
        preprocessor.preprocess_metadata(
            metadata,
            alert_number=0,
        )


def test_legacy_sample_does_not_propagate_label():
    preprocessor = ZTFAstroMiNNPreprocessor()

    sample = {
        "obj_id": "ZTF17aaaemzh",
        "alerte": 0,
        "images": np.zeros(
            (63, 63, 3),
            dtype=np.float32,
        ),
        "metadata": historical_metadata(),
        "target": "Cataclysmic",
    }

    result = preprocessor.preprocess_legacy_sample(sample)

    assert result.object_id == "ZTF17aaaemzh"
    assert result.alert == 0
    assert not hasattr(result, "target")
