"""Tests for binMd -- the outcome-first front-end over binmed."""

from morie.fn import _array_core as np

from morie.fn.binmed import binary_outcome_mediation as canonical
from morie.fn.binMd import binary_outcome_mediation as frontend


def _data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n).astype(float)
    m = 1.0 * x + rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.5 * x + 1.0 * m)))).astype(float)
    return x, m, y


def test_front_end_matches_the_canonical_implementation():
    """Same numbers, only the argument order differs."""
    x, m, y = _data(seed=1)
    a = canonical(x, m, y)
    b = frontend(y, x, m)
    for k in ("total", "direct", "indirect", "or_indirect"):
        assert a[k] == b[k]


def test_it_does_not_carry_a_second_implementation():
    import inspect

    from morie.fn import binMd

    assert "binmed" in inspect.getsource(binMd)
    assert len(inspect.getsource(binMd.binary_outcome_mediation).splitlines()) < 25


def test_covariates_pass_through():
    x, m, y = _data(seed=2)
    c = np.random.default_rng(2).normal(0, 1, (x.size, 2))
    assert frontend(y, x, m, C=c)["indirect"] == canonical(x, m, y, C=c)["indirect"]
