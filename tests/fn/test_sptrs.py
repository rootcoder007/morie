"""Tests for sptrs.schabenberger_trend_surface.

sptrs delegates to sptrn.spatial_trend_surface; the contract these tests pin is that
the delegation reaches the same estimator with the arguments mapped
correctly, so the two must agree exactly.
"""

from morie.fn import _array_core as np

from morie.fn.sptrs import schabenberger_trend_surface
from morie.fn.sptrn import spatial_trend_surface


def _same(a, b):
    """Deep-compare two results.

    ``RichResult`` is a dict subclass whose ``__str__`` is empty unless it was
    built with a title, so comparing ``str(a) == str(b)`` passes vacuously.
    Compare the payload itself, array-aware.
    """
    from morie.fn import _array_core as np

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
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1, (60, 2))
    z = 3.0 * coords[:, 0] - 2.0 * coords[:, 1] + rng.normal(0, 0.1, 60)
    return coords, z


def test_sptrs_matches_sptrn():
    """The delegation must reproduce the implemented estimator exactly."""
    coords, z = _fixture()
    got = schabenberger_trend_surface(coords, z, 2)
    ref = spatial_trend_surface(z, coords, order=2)
    assert _same(got, ref)
    assert dict(got)  # not an empty payload


def test_sptrs_edge():
    """Degenerate input is handled by the implemented estimator, not here."""
    coords, z = _fixture()
    # a first-order surface is a different fit, not the same one
    linear = schabenberger_trend_surface(coords, z, 1)
    assert not _same(linear, schabenberger_trend_surface(coords, z, 2))
