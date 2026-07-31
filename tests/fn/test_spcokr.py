"""Tests for spcokr.schabenberger_cokriging.

spcokr delegates to cokrg.cokriging; the contract these tests pin is that
the delegation reaches the same estimator with the arguments mapped
correctly, so the two must agree exactly.
"""

import numpy as np

from morie.fn.spcokr import schabenberger_cokriging
from morie.fn.cokrg import cokriging


def _same(a, b):
    """Deep-compare two results.

    ``RichResult`` is a dict subclass whose ``__str__`` is empty unless it was
    built with a title, so comparing ``str(a) == str(b)`` passes vacuously.
    Compare the payload itself, array-aware.
    """
    import numpy as np

    if isinstance(a, dict) != isinstance(b, dict):
        return False
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)):
        a_arr, b_arr = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        if a_arr.shape != b_arr.shape:
            return False
        return bool(np.all([_same(x, y) for x, y in
                            zip(a_arr.ravel(), b_arr.ravel())])) if a_arr.size else True
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return a == b


def _fixture():
    rng = np.random.default_rng(5)
    coords = rng.random((30, 2))
    z1 = rng.normal(size=30)
    z2 = 0.7 * z1 + rng.normal(0, 0.3, 30)
    target = np.array([[0.5, 0.5]])
    return coords, z1, z2, target


def test_spcokr_matches_cokrg():
    """The delegation must reproduce the implemented estimator exactly."""
    coords, z1, z2, target = _fixture()
    got = schabenberger_cokriging(
        coords, z1, z2, target, {"sill_p": 1.0, "range_p": 0.3}
    )
    ref = cokriging(z1, z2, coords, target, sill_p=1.0, range_p=0.3)
    assert _same(got, ref)
    assert dict(got)  # not an empty payload


def test_spcokr_edge():
    """Degenerate input is handled by the implemented estimator, not here."""
    coords, z1, z2, target = _fixture()
    # primary and secondary are not interchangeable
    a = schabenberger_cokriging(coords, z1, z2, target)
    b = schabenberger_cokriging(coords, z2, z1, target)
    assert not _same(a, b)
