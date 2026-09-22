"""Tests for gh_c6_14.ghosal_pred_consist."""

from morie.fn import _array_core as np

from morie.fn.gh_c6_14 import ghosal_pred_consist


def test_gh_c6_14_basic():
    """Test basic functionality with documented signature."""
    result = ghosal_pred_consist()
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    assert "cesaro_kl_path" in result
    assert "decaying" in result
    assert "method" in result
    # Documented behaviour: Cesaro-averaged KL to the truth decays.
    assert result["decaying"] is True
    # Independent check: path[-1] equals estimate.
    assert result["estimate"] == result["cesaro_kl_path"][-1]


def test_gh_c6_14_path_arithmetic():
    """Documented inputs reproduce the documented formula's behaviour.

    With theta0 = 0.35, n = 800, seed = 42, the predictive estimate before
    the i-th observation is pred_i = (1 + S_i) / (2 + i), where S_i is the
    number of successes in the first i Bernoulli(theta0) draws. The i-th
    incremental KL is

        kl_i = theta0 * log(theta0 / pred_i)
             + (1 - theta0) * log((1 - theta0) / (1 - pred_i))

    The returned estimate is the Cesaro mean of these KL values over the
    last block of i's (those with (i+1) % (n // 8) == 0).
    """
    import math

    theta0 = 0.35
    n = 800
    seed = 42
    result = ghosal_pred_consist(theta0=theta0, n=n, seed=seed)

    path = result["cesaro_kl_path"]
    block = n // 8  # sampling cadence for the path entries

    # Reproduce the draws and the KL accumulation independently.
    rng = np.random.default_rng(seed)
    S = 0
    tot_kl = 0.0
    expected_path = []
    for i in range(n):
        pred = (1.0 + S) / (2.0 + i)
        tot_kl += theta0 * math.log(theta0 / pred) \
            + (1 - theta0) * math.log((1 - theta0) / (1 - pred))
        if (i + 1) % block == 0:
            expected_path.append(tot_kl / (i + 1))
        u = float(rng.uniform(0, 1))
        S += 1 if u < theta0 else 0

    # Last path entry must match the documented estimate (not a copied number).
    assert np.allclose(
        np.asarray(path[-1], dtype=float),
        np.asarray(expected_path[-1], dtype=float),
    )
    # Documented decay: Cesaro mean at the end is below the first one.
    assert expected_path[-1] < expected_path[0]
    assert result["decaying"] is True
    # Estimate and path tail agree.
    assert np.allclose(
        np.asarray(result["estimate"], dtype=float),
        np.asarray(expected_path[-1], dtype=float),
    )
