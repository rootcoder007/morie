"""Test opall."""

from morie.fn import _array_core as np

from morie.fn.opall import opall


def test_opall_basic():
    rng = np.random.default_rng(42)
    r = opall(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opall_description():
    rng = np.random.default_rng(42)
    r = opall(n_dims=2, max_iter=50)
    assert r.name
