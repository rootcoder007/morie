"""Tests for blupr.blup_random_intercept."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.blupr import blup_random_intercept


def test_blupr_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    rng_g = np.random.default_rng(41)
    rng_u = np.random.default_rng(40)
    rng_e = np.random.default_rng(39)

    y = rng_y.normal(0.0, 1.0, 100)
    X = rng_x.normal(0.0, 1.0, (100, 5))
    # Cluster labels must be integers in {0, 1, ..., 9}, not continuous draws.
    cluster = rng_g.integers(0, 10, size=100)
    sigma2_u = float(rng_u.uniform(0.5, 1.5))
    sigma2_e = float(rng_e.uniform(0.5, 1.5))

    result = blup_random_intercept(y, cluster, sigma2_u, sigma2_e,
                                   X=X, beta=np.zeros(5))

    # The function returns a RichResult (dict-like) with documented keys.
    assert isinstance(result, dict)
    for key in ("u", "shrink", "nj", "groupmean", "levels",
                "vpc", "J", "n"):
        assert key in result

    # Shape/scalar sanity on the returned object.
    assert result["n"] == 100
    assert result["J"] == 10
    assert len(result["levels"]) == 10
    assert len(result["u"]) == 10
    assert len(result["shrink"]) == 10
    assert len(result["nj"]) == 10
    assert len(result["groupmean"]) == 10

    # Variance partitioning coefficient lies in [0, 1].
    assert 0.0 <= result["vpc"] <= 1.0
    # Each shrinkage factor lies in [0, 1].
    for k in result["shrink"]:
        assert 0.0 <= k <= 1.0
    # Group sizes sum to n.
    assert int(sum(result["nj"])) == 100

    # Independent recomputation of the shrinkage factors and BLUPs
    # from the documented formula, using raw Python arithmetic on the
    # same inputs.
    n = len(y)
    g = list(cluster)
    s2u = float(sigma2_u)
    s2e = float(sigma2_e)

    # Residuals with a zero fixed-effect coefficient vector.
    resid = [float(y[i]) - 0.0 for i in range(n)]

    # Reproduce the function's level ordering (first appearance wins).
    labs = []
    for v in g:
        if v not in labs:
            labs.append(v)

    expected_u = []
    expected_k = []
    expected_gm = []
    for L in labs:
        idx = [i for i in range(n) if g[i] == L]
        m = len(idx)
        mean = sum(resid[i] for i in idx) / m
        k = s2u / (s2u + s2e / m)
        expected_gm.append(mean)
        expected_k.append(k)
        expected_u.append(k * mean)

    assert list(result["levels"]) == labs
    assert list(result["groupmean"]) == expected_gm
    assert list(result["shrink"]) == expected_k
    assert list(result["u"]) == expected_u

    # Documented asymptotic behaviour: complete pooling when s2u == 0.
    r0 = blup_random_intercept(y, cluster, 0.0, sigma2_e,
                               X=X, beta=np.zeros(5))
    assert list(r0["u"]) == [0.0] * 10
    assert list(r0["shrink"]) == [0.0] * 10


def test_blupr_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_x = np.random.default_rng(42)
    rng_g = np.random.default_rng(41)
    rng_u = np.random.default_rng(40)
    rng_e = np.random.default_rng(39)

    y = rng_y.normal(0.0, 1.0, 100)
    X = rng_x.normal(0.0, 1.0, (100, 5))
    cluster = rng_g.integers(0, 10, size=100)
    sigma2_u = float(rng_u.uniform(0.5, 1.5))
    sigma2_e = float(rng_e.uniform(0.5, 1.5))

    # No fixed effects: X=None, beta=None; raw response is used.
    result = blup_random_intercept(y, cluster, sigma2_u, sigma2_e)

    assert isinstance(result, dict)
    assert result["n"] == 100
    assert result["J"] == 10
    for key in ("u", "shrink", "nj", "groupmean", "levels", "vpc"):
        assert key in result

    # Complete pooling still yields all-zero random effects.
    r0 = blup_random_intercept(y, cluster, 0.0, sigma2_e)
    assert list(r0["u"]) == [0.0] * result["J"]
