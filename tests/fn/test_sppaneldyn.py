"""Tests for sppaneldyn: the dynamic spatial panel is the FE spatial lag model on augmented regressors."""

import math

from morie.fn.sppanel import spatial_panel_ml
from morie.fn.sppaneldyn import sp_panel_dynamic

N, T = 6, 5
_A = [[1.0 if (i != j and abs(i - j) <= 1) else 0.0 for j in range(N)] for i in range(N)]
W = [[a / sum(r) for a in r] for r in _A]
X = [[[math.sin(i + 2 * t), 0.3 * math.cos(i * t)] for i in range(N)] for t in range(T)]
Y = [
    [0.6 * X[t][i][0] - 0.4 * X[t][i][1] + 0.5 * math.sin(i * 1.3) + 0.2 * math.cos(t + i) for i in range(N)]
    for t in range(T)
]


def _augmented(space_time_lag):
    yv, Xm = [], []
    for t in range(1, T):
        prev = Y[t - 1]
        wprev = [sum(W[i][j] * prev[j] for j in range(N)) for i in range(N)]
        for i in range(N):
            yv.append(Y[t][i])
            Xm.append([prev[i]] + ([wprev[i]] if space_time_lag else []) + list(X[t][i]))
    return yv, Xm


def test_dynamic_panel_equals_fe_lag_on_the_augmented_design():
    for stl in (True, False):
        r = sp_panel_dynamic(Y, X, W, space_time_lag=stl)
        yv, Xm = _augmented(stl)
        ref = spatial_panel_ml(yv, Xm, W, N, model="lag", effects="individual")
        assert len(r.beta) == (4 if stl else 3)
        assert r.beta == ref.coefficients and r.rho == ref.rho and r.loglik == ref.loglik
        assert r.n_obs == N * (T - 1) and len(r.residuals) == N * (T - 1)


def test_dynamic_panel_twoways_effects_and_bounds_pass_through():
    r = sp_panel_dynamic(Y, X, W, effects="twoways", bounds=(-0.5, 0.5))
    yv, Xm = _augmented(True)
    ref = spatial_panel_ml(yv, Xm, W, N, model="lag", effects="twoways", interval=(-0.5, 0.5))
    assert r.rho == ref.rho and r.sigma2 == ref.sigma2
    assert -0.5 <= r.rho <= 0.5
