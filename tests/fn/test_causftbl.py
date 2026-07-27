"""Tests for causftbl.causal_frontdoor_adjustment."""

import numpy as np
import pytest

from morie.fn.causftbl import causal_frontdoor_adjustment

P_Z_X = np.array([[0.9, 0.1], [0.2, 0.8]])
P_Y_XZ = np.array([[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.1, 0.9]]])
P_X = np.array([0.5, 0.5])


def test_causftbl_basic():
    out = causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, P_X)
    assert out["p_y_do_x"].sum(axis=1) == pytest.approx([1.0, 1.0])
    inner1 = np.array([0.5 * 0.2 + 0.5 * 0.4, 0.5 * 0.7 + 0.5 * 0.9])
    assert out["p_y_do_x"][0, 1] == pytest.approx(0.9 * inner1[0] + 0.1 * inner1[1])


def test_causftbl_edge():
    with pytest.raises(ValueError):
        causal_frontdoor_adjustment([[0.5, 0.9]], P_Y_XZ[:1], [1.0])  # rows do not sum to 1
    with pytest.raises(ValueError):
        causal_frontdoor_adjustment(P_Z_X, P_Y_XZ, [0.5, 0.9])  # P_X does not sum to 1
