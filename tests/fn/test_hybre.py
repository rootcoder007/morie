"""Test hybre."""

from morie.fn import _array_core as np

from morie.fn.hybre import hybre


def test_hybre_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hybre(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hybre_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hybre(flow=flow, precip=precip, n=20)
    assert r.name
