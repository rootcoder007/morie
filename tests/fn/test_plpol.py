"""Tests for plpol."""

import numpy as np
import pytest

from morie.fn.plpol import plot_spatial


def test_plpol_basic():
    out = plot_spatial(np.array([[-1.0, 0.0], [1.0, 0.0]]), party_labels=["D", "R"])
    assert out["centroids"]["D"] == pytest.approx([-1.0, 0.0])
    assert out["coords"].shape == (2, 2)


def test_plpol_edge():
    assert plot_spatial([0.0, 1.0])["coords"].shape == (2, 2)  # 1-D padded
    with pytest.raises(ValueError):
        plot_spatial([[0.0, 1.0]], party_labels=["D", "R"])  # label mismatch
