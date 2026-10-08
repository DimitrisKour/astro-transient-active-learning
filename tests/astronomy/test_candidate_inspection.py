"""Candidate plotting remains useful after provider metadata removal."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from al_pipeline.astronomy.astrominn import inspection


def test_plot_accepts_caller_supplied_metadata_without_network(tmp_path, monkeypatch):
    path = tmp_path / "candidate.npy"
    np.save(path, {"image": np.zeros((3, 8, 8)), "metadata": np.zeros(34)})
    frame = pd.DataFrame(
        [
            {
                "obj_id": "candidate-1",
                "alert": 2,
                "acquisition_score": 0.7,
                "filepath": str(path),
                "source_reference": "boom:1",
                "candid": "1",
                "drb": 0.9,
            }
        ]
    )
    monkeypatch.setattr(inspection, "load_astrominn_vit_config", lambda: {})
    monkeypatch.setattr(
        inspection,
        "preprocess_astrominn_vit",
        lambda **kw: (torch.zeros(34), torch.tensor(kw["image"])),
    )
    original = frame.copy(deep=True)
    figures = inspection.plot_candidate_table(frame, metadata_fields=("drb",), show=False)
    try:
        assert len(figures) == 1
        assert any("drb: 0.9" in text.get_text() for ax in figures[0].axes for text in ax.texts)
        assert len(figures[0].axes) == 7
        pd.testing.assert_frame_equal(frame, original)
    finally:
        for figure in figures:
            plt.close(figure)
