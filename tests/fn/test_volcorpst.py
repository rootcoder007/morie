"""Tests for volcorpst.vol_corradi_swan_persistence."""

import numpy as np
import pytest
from scipy import stats

from morie.fn.volcorpst import vol_corradi_swan_persistence


def _garch(seed, n=4000, omega=0.05, alpha=0.25, beta=0.72):
    rng = np.random.default_rng(seed)
    s2, y = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        y[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * y[t] ** 2 + beta * s2
    return y


def test_volcorpst_reports_every_requested_horizon():
    rng = np.random.default_rng(0)
    r = vol_corradi_swan_persistence(rng.standard_normal(800), horizons=(1, 5, 20), n_mc=200)
    per = r["per_horizon"]
    assert [e["h"] for e in per] == [1, 5, 20]
    assert per[0]["n_h"] == 800 and per[1]["n_h"] == 160 and per[2]["n_h"] == 40
    assert float(r["statistic"]) == pytest.approx(max(e["statistic"] for e in per), rel=1e-12)


def test_volcorpst_gaussian_data_passes_at_all_horizons():
    """Aggregates of i.i.d. Gaussians are Gaussian at every horizon, so
    the fitted-normal test should not reject. Measured joint p > 0.2 on
    seeds 0..2."""
    for s in range(3):
        rng = np.random.default_rng(s)
        r = vol_corradi_swan_persistence(rng.standard_normal(1200), horizons=(1, 5, 10), n_mc=200, seed=s)
        assert float(r["p_value"]) > 0.05


def test_volcorpst_detects_heavy_tails_at_short_horizons():
    """i.i.d. t(3) is far from Gaussian at h = 1 but its 20-period
    aggregates are already CLT-normalised: rejection must come from the
    SHORT horizon. This is what checking multiple horizons is for."""
    rng = np.random.default_rng(1)
    r = vol_corradi_swan_persistence(rng.standard_t(3, 3000), horizons=(1, 20), n_mc=300)
    per = {e["h"]: e for e in r["per_horizon"]}
    assert per[1]["p_value"] < 0.01
    assert per[1]["statistic"] > per[20]["statistic"]
    assert float(r["p_value"]) < 0.05


def test_volcorpst_specified_cdf_uses_the_exact_null():
    """With the true CDF supplied there is no estimation, and the classical
    Kolmogorov distribution applies: p must be well-behaved under the
    null and reject under a wrong CDF."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(600)
    ok = vol_corradi_swan_persistence(
        x, horizons=(1, 4), cdf=lambda v, h: stats.norm.cdf(v, scale=np.sqrt(h))
    )
    assert float(ok["p_value"]) > 0.05
    bad = vol_corradi_swan_persistence(
        x, horizons=(1, 4), cdf=lambda v, h: stats.norm.cdf(v, scale=3.0 * np.sqrt(h))
    )
    assert float(bad["p_value"]) < 0.01


def test_volcorpst_rejects_bad_input():
    with pytest.raises(ValueError, match="positive"):
        vol_corradi_swan_persistence(np.arange(100.0), horizons=(0, 5))
    with pytest.raises(ValueError, match="at least 8 aggregates"):
        vol_corradi_swan_persistence(np.arange(50.0), horizons=(20,))
