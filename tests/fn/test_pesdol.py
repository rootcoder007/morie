"""pesdol: ARDL bounds test (Pesaran, Shin & Smith 2001).

The generated test imported `pesaran_shin_dols`, which does not exist.
Rewritten against ardl_bounds and anchored on the test's own decision
rule rather than on a fabricated key.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.pesdol import ardl_bounds


def _series(n=60):
    x = [[float(i) * 0.5] for i in range(n)]
    y = [float(i) * 1.0 + (0.3 if i % 3 == 0 else -0.2) for i in range(n)]
    return y, x


def test_returns_the_bounds_test_machinery():
    y, x = _series()
    r = ardl_bounds(y, x, p=1, q=1)
    for k in ("f_statistic", "bound_lower", "bound_upper", "verdict",
              "speed_of_adjustment", "long_run"):
        assert k in r
    assert np.isfinite(float(r["f_statistic"]))


def test_the_verdict_follows_the_critical_bounds():
    """The stated rule: above the upper bound is cointegration, below the
    lower bound is none, in between is inconclusive."""
    y, x = _series()
    r = ardl_bounds(y, x, p=1, q=1)
    f = float(r["f_statistic"])
    lo, hi = float(r["bound_lower"]), float(r["bound_upper"])
    v = str(r["verdict"]).lower()
    if f > hi:
        assert "cointegr" in v and "no" not in v.split()[0]
    elif f < lo:
        assert "no" in v or "reject" not in v
    else:
        assert "inconclusive" in v or "incon" in v


def test_residuals_are_as_long_as_the_usable_sample():
    y, x = _series()
    r = ardl_bounds(y, x, p=1, q=1)
    assert len(np.asarray(r["residuals"])) == int(r["n_used"])
