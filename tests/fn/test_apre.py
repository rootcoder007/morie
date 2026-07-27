"""Tests for apre."""

import numpy as np
import pytest

from morie.fn.apre import oc_apre


def test_apre_basic():
    obs = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=float).T
    pred = obs.copy()
    pred[6, 0] = 1
    assert oc_apre(obs, pred)["apre"] == pytest.approx(0.75)


def test_apre_edge():
    with pytest.raises(ValueError):
        oc_apre(np.ones((4, 1)), np.ones((4, 1)))  # unanimous roll call
    with pytest.raises(ValueError):
        oc_apre(np.ones((4, 1)), np.ones((3, 1)))  # shape mismatch
