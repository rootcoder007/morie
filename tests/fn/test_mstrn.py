"""Tests for mstrn."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mstrn import multistate_transition_matrix

_TIME = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_FROM = np.array([0, 0, 0, 1, 0, 1])
_TO = np.array([1, 1, 2, 2, 1, 2])


def test_mstrn_basic():
    out = multistate_transition_matrix(_TIME, _FROM, _TO, n_states=3)
    assert out["P"].shape == (3, 3)
    assert np.allclose(out["P"].sum(axis=1), 1.0)  # rows are probabilities


def test_mstrn_edge():
    out = multistate_transition_matrix(_TIME, _FROM, _TO, n_states=3)
    for dA in out["increments"]:
        assert np.allclose(dA.sum(axis=1), 0.0)  # zero row sums by construction
    with pytest.raises(ValueError):
        multistate_transition_matrix(_TIME, _FROM, _TO, n_states=2)
