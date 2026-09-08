"""Tests for bfsd.savage_dickey_ratio (Savage-Dickey Bayes factor)."""

import math

from morie.fn import _array_core as np
from morie.fn.bfsd import savage_dickey_ratio


def _prior_n04(t):
    return math.exp(-t * t / 8.0) / math.sqrt(8.0 * math.pi)


def test_bfsd_posterior_far_from_null_gives_small_bf01():
    # samples centred at 3: posterior density at 0 is tiny -> BF01 small,
    # BF10 large (evidence against theta0 = 0)
    rng = np.random.default_rng(11)
    far = rng.normal(3.0, 0.5, 500)
    r = savage_dickey_ratio(far, _prior_n04, theta0=0.0)
    assert r["estimate"] < 0.5
    assert r["bf10"] > 2.0
    assert abs(r["estimate"] * r["bf10"] - 1.0) < 1e-9


def test_bfsd_ratio_definition():
    rng = np.random.default_rng(12)
    s = rng.normal(0.0, 1.0, 500)
    r = savage_dickey_ratio(s, _prior_n04, theta0=0.0)
    want = r["posterior_density_at_null"] / r["prior_density_at_null"]
    assert abs(r["estimate"] - want) < 1e-12
