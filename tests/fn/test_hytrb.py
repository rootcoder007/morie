"""Test hytrb."""

from morie.fn import _array_core as np

from morie.fn.hytrb import hytrb


def test_hytrb_basic():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hytrb(flow=flow, precip=precip, n=20)
    assert r.value is not None


def test_hytrb_description():
    rng = np.random.default_rng(42)
    flow = np.abs(rng.standard_normal(20)) * 100
    precip = np.abs(rng.standard_normal(20)) * 50
    r = hytrb(flow=flow, precip=precip, n=20)
    assert r.name
