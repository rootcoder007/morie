"""Tests for spblkk.schabenberger_block_kriging.

spblkk delegates to spblk.spatial_block_kriging; the contract these tests pin is that
the delegation reaches the same estimator with the arguments mapped
correctly, so the two must agree exactly.
"""

from morie.fn import _array_core as np

from morie.fn.spblkk import schabenberger_block_kriging
from morie.fn.spblk import spatial_block_kriging


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
    rng = np.random.default_rng(3)
    coords = rng.random((30, 2))
    z = coords[:, 0] * 2.0 + rng.normal(0, 0.2, 30)
    blocks = [np.array([[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]])]
    return coords, z, blocks


def test_spblkk_matches_spblk():
    """The delegation must reproduce the implemented estimator exactly."""
    coords, z, blocks = _fixture()
    got = schabenberger_block_kriging(
        coords, z, blocks, {"nugget": 0.1, "sill": 1.0, "range": 0.3}
    )
    ref = spatial_block_kriging(z, coords, blocks, nugget=0.1, sill=1.0, range_=0.3)
    assert _same(got, ref)
    assert dict(got)  # not an empty payload


def test_spblkk_edge():
    """Degenerate input is handled by the implemented estimator, not here."""
    coords, z, blocks = _fixture()
    # cov_model is optional and "range" is accepted as an alias for range_
    assert _same(
        schabenberger_block_kriging(coords, z, blocks, {"range": 0.3}),
        spatial_block_kriging(z, coords, blocks, range_=0.3),
    )
    assert schabenberger_block_kriging(coords, z, blocks) is not None
