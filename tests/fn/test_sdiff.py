"""Tests for sdiff.synthetic_did (Arkhangelsky et al. 2021, Algorithms 1 and 3)."""

import math
import statistics

import pytest

from morie.fn.sdiff import synthetic_did


N, TT, T0 = 24, 12, 8
TR = [20, 21, 22, 23]
F = [sum(math.sin(1.3 * k) for k in range(t + 1)) for t in range(TT)]
LOAD = [0.4 + 1.2 * ((i * 0.618) % 1) for i in range(N)]
Y = [[LOAD[i] * F[t] + 0.5 * math.sin(12.9898 * i * i + 78.233 * t * t + 3.1 * i * t)
      + ((2.0 + 0.3 * i) if i in TR and t >= T0 else 0.0) for t in range(TT)] for i in range(N)]


def _eff(i, lam):
    return sum(Y[i][T0:]) / (TT - T0) - sum(l * y for l, y in zip(lam, Y[i][:T0]))


def test_sdiff_basic():
    """tau = sum over treated of (post mean - lambda'pre) / N_tr minus the
    omega-weighted same for controls; zeta = (N_tr T_post)^{1/4} x the sd
    of control first differences; the jackknife holds omega and lambda
    fixed (omega renormalised); values agree with the authors' synthdid
    (sparsify off, zeta_lambda = 0): tau 8.65707738, jackknife se
    0.39334990, to its Frank-Wolfe stopping precision."""
    r = synthetic_did(Y, None, None, TR, T0)
    w = [float(v) for v in r["unit_weights"]]
    lam = [float(v) for v in r["time_weights"]]
    ctrl = [i for i in range(N) if i not in TR]
    tau = sum(_eff(i, lam) for i in TR) / 4 - sum(wj * _eff(i, lam) for wj, i in zip(w, ctrl))
    assert r["estimate"] == pytest.approx(tau, rel=1e-12)
    d = [Y[i][t + 1] - Y[i][t] for i in ctrl for t in range(T0 - 1)]
    assert r["zeta"] == pytest.approx((4 * (TT - T0)) ** 0.25 * statistics.stdev(d), rel=1e-12)
    jk = []
    for out in TR + ctrl:
        wk = [0.0 if i == out else wj for wj, i in zip(w, ctrl)]
        s = sum(wk)
        tk = [i for i in TR if i != out]
        jk.append(sum(_eff(i, lam) for i in tk) / len(tk) - sum(v / s * _eff(i, lam) for v, i in zip(wk, ctrl)))
    m = sum(jk) / N
    assert r["se"] == pytest.approx(math.sqrt((N - 1) / N * sum((v - m) ** 2 for v in jk)), rel=1e-12)
    # synthdid's Frank-Wolfe stops at min.decrease, so it matches to ~1e-6
    assert r["estimate"] == pytest.approx(8.65707737969762, rel=1e-6)
    assert r["se"] == pytest.approx(0.393349898310404, rel=1e-5)


def test_sdiff_edge():
    """Fewer than two controls or fewer than two pre-periods raise; one
    treated unit leaves the jackknife undefined."""
    with pytest.raises(ValueError):
        synthetic_did([row for i, row in enumerate(Y) if i in TR or i < 1], None, None, [1, 2, 3, 4], T0)
    with pytest.raises(ValueError):
        synthetic_did(Y, None, None, TR, 1)
    assert math.isnan(synthetic_did(Y, None, None, [23], T0)["se"])
