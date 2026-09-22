"""Tests for gh_c12_1.ghosal_infdim_bvm."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c12_1 import ghosal_infdim_bvm


def test_gh_c12_1_basic():
    """Test basic functionality with default args and a fixed seed."""
    result = ghosal_infdim_bvm(theta0=0.4, n=2000, seed=42)
    assert "estimate" in result
    est = float(np.asarray(result["estimate"], dtype=float))
    assert math.isfinite(est)
    assert 0.0 <= est < 0.05
    assert result["bvm_holds"] is True
    assert result["method"].startswith("parametric BvM")


def test_gh_c12_1_seeded_reproducible():
    """Same seed, same args -> same total-variation estimate."""
    r1 = ghosal_infdim_bvm(theta0=0.4, n=2000, seed=42)
    r2 = ghosal_infdim_bvm(theta0=0.4, n=2000, seed=42)
    assert float(r1["estimate"]) == float(r2["estimate"])


def test_gh_c12_1_independent_total_variation():
    """Compare against an independent Riemann-sum implementation of the
    formula documented in the docstring (Ghosal / v.d. Vaart sec. 12.1):
    TV = 0.5 * sum |Beta(a,b)(t) - N(mle, I_inv/n)(t)| * dh."""
    theta0, n, seed = 0.3, 1500, 7

    rng = np.random.default_rng(seed)
    S = 0
    for _ in range(n):
        if float(rng.uniform(0, 1)) < theta0:
            S += 1

    a = 1.0 + S
    b = 1.0 + n - S
    mle = S / n
    I_inv = mle * (1.0 - mle)
    sd = math.sqrt(I_inv / n)
    grid = 2000
    lo = max(mle - 6 * sd, 1e-9)
    hi = min(mle + 6 * sd, 1.0 - 1e-9)
    expected = 0.0
    for i in range(grid):
        t = lo + (hi - lo) * (i + 0.5) / grid
        bpdf = math.exp(
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + (a - 1) * math.log(t) + (b - 1) * math.log(1.0 - t)
        )
        npdf = math.exp(-0.5 * ((t - mle) / sd) ** 2) / (
            sd * math.sqrt(2.0 * math.pi)
        )
        expected += 0.5 * abs(bpdf - npdf) * (hi - lo) / grid

    result = ghosal_infdim_bvm(theta0=theta0, n=n, seed=seed)
    est = float(np.asarray(result["estimate"], dtype=float))
    assert math.isclose(est, expected, rel_tol=1e-6, abs_tol=1e-9)
    assert result["bvm_holds"] == (expected < 0.05)
