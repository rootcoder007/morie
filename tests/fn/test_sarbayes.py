"""sarbayes: sampler building blocks against their definitions, reproducibility and support."""

import math

import pytest

from morie.fn._rrng_core import pnorm
from morie.fn.sarbayes import (
    _draw_rho,
    _logdet_grid,
    _logdet_lu,
    _tnorm,
    sar_ordered_probit_gibbs,
    sar_probit_gibbs,
    sar_tobit_gibbs,
    spatial_bayes_gibbs,
)

W = [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0.5, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0, 1, 0]]
X = [[1.0, v] for v in (1.2, 0.4, -0.3, -0.9, 0.1)]


def test_truncated_normal_is_the_inverse_cdf():
    for m, s, lo, hi, u in [
        (0.3, 1.2, 0.0, math.inf, 0.37),
        (-2.0, 0.5, 0.0, math.inf, 0.9),
        (1.0, 2.0, -math.inf, 0.0, 0.2),
        (0.0, 1.0, -0.5, 1.5, 0.61),
    ]:
        x = _tnorm(m, s, lo, hi, u)
        assert lo <= x <= hi
        fa = pnorm((lo - m) / s) if lo > -math.inf else 0.0
        fb = pnorm((hi - m) / s) if hi < math.inf else 1.0
        # the lower-bounded case maps u through the upper tail (1 - u) for accuracy
        want = 1 - u if hi == math.inf and lo > -math.inf else u
        assert (pnorm((x - m) / s) - fa) / (fb - fa) == pytest.approx(want, rel=1e-9)


def test_logdet_grid_matches_lu():
    grid = [-0.8, -0.1, 0.35, 0.9]
    exact = [_logdet_lu([[(1.0 if i == j else 0.0) - r * W[i][j] for j in range(5)] for i in range(5)]) for r in grid]
    assert _logdet_grid(W, grid) == pytest.approx(exact, rel=1e-12)
    # a W without the symmetric-similar structure uses the interpolated grid
    W2 = [[0, 0.7, 0.3], [0.2, 0, 0.8], [0.5, 0.5, 0]]
    ex2 = [_logdet_lu([[(1.0 if i == j else 0.0) - r * W2[i][j] for j in range(3)] for i in range(3)]) for r in grid]
    assert _logdet_grid(W2, grid) == pytest.approx(ex2, abs=2e-4)


def test_draw_rho_inverse_cdf():
    grid = [-0.5, 0.0, 0.5]
    lnd = [0.0, 0.0, 0.0]
    lp = [math.log(1.0), math.log(2.0), math.log(1.0)]
    assert _draw_rho(grid, lnd, lp, 0.0, 0.0, 0.0, 0.0, 0.2) == -0.5
    assert _draw_rho(grid, lnd, lp, 0.0, 0.0, 0.0, 0.0, 0.5) == 0.0
    assert _draw_rho(grid, lnd, lp, 0.0, 0.0, 0.0, 0.0, 0.9) == 0.5


def test_samplers_reproducible_and_supported():
    y = [1, 1, 0, 0, 1]
    a = sar_probit_gibbs(y, X, W, ndraw=30, burn_in=5, seed=4)
    b = sar_probit_gibbs(y, X, W, ndraw=30, burn_in=5, seed=4)
    assert a.rho_draws == b.rho_draws and all(-1 < r < 1 for r in a.rho_draws)
    t = sar_tobit_gibbs([1.3, 0.2, 0.0, 0.0, 0.7], X, W, ndraw=30, burn_in=5, seed=1)
    assert t.sigma2 > 0
    o = sar_ordered_probit_gibbs([3, 2, 1, 1, 2], X, W, ndraw=30, burn_in=5, seed=1)
    assert o.cutpoints[0] == 0.0 and o.cutpoints[1] > 0
    for model in ("lag", "error", "durbin"):
        r = spatial_bayes_gibbs([2.0, 1.1, 0.2, -0.8, 0.9], X, W, model, ndraw=30, burn_in=5, seed=2)
        assert len(r.beta) == (3 if model == "durbin" else 2)
    with pytest.raises(ValueError):
        sar_probit_gibbs(y, X, W, ndraw=5, burn_in=1, method="gibbs")
