"""Schabenberger & Gotway Sec. 4.6 and Problem 2.2 methods against R references."""

import math

from morie.fn.cndchk import semivariogram_cnd_check
from morie.fn.nnlsq import nonnegative_least_squares
from morie.fn.spkrnv import kernel_semivariogram
from morie.fn.spnpsv import shapiro_botha_semivariogram
from morie.fn.zegcnt import zeger_count_moments

H = [float(k) for k in range(1, 9)]


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_zeger_book_example():
    # p. 294: sigma2 = 1 and mu(s) = mu(s + h) = 1 give rho1 / 2
    assert close(zeger_count_moments(1, 1, 1, 0.6)["corr"], 0.3, 1e-15)
    r = zeger_count_moments(2.0, 0.5, 0.8, 0.9)
    assert close(r["corr"], 0.9 / math.sqrt((1 + 1 / 1.6) * (1 + 1 / 0.4)), 1e-15)


def test_nnls_matches_r_nnls():
    A = [[math.sin(1.3 * i + 0.7 * j) + 0.1 * (i == j) for j in range(4)] for i in range(6)]
    b = [math.cos(0.9 * i) for i in range(6)]
    r = nonnegative_least_squares(A, b)
    # nnls::nnls(A, b)
    for a, e in zip(r["x"], [0.96830801856704418, 0.0, 0.0, 0.41394598214996808]):
        assert close(a, e, 1e-13)
    assert close(r["residual_norm"], 1.0687783699685953, 1e-13)


def test_shapiro_botha_fit_matches_nnls_on_j0_basis():
    g = [1.1 - 1.1 * math.exp(-k / 3) + 0.04 * math.sin(2 * k) for k in range(1, 9)]
    r = shapiro_botha_semivariogram(H, [0.1, 0.375, 0.65, 0.925, 1.2], gamma_hat=g, d=2)
    # nnls::nnls(1 - besselJ(outer(h, t), 0), g)
    ref = [0.0, 0.23583024764001043, 0.34227825940611956, 0.0, 0.30395031089466884]
    for a, e in zip(r["weights"], ref):
        assert close(a, e, 1e-12)
    assert close(r["rss"], 0.051067101651968011, 1e-12)


def test_kernel_covariance_matches_integrate_besselj():
    h = [0, 1, 5, 12, 30]
    # 1.5 * (G(0) + (1 - G(b)) J0(h b) + integrate(besselJ(h w, 0), lo, hi) / width)
    cases = {
        (0.1, 0.3, 1.0): [1.5, 1.483806617205643, 1.1276535116158839, 0.07569893717565343, -0.03382520149877942],
        (-0.1, 0.1, 1.0): [1.5, 1.4993752343285029, 1.4845207599690675, 1.4147235444680419, 1.0968918130024661],
        (0.1, 0.4, 0.25): [1.5, 1.4822482806342059, 1.0912352688918874, -0.060034974151034377, 0.15168877044325332],
    }
    for (tl, tu, b), ref in cases.items():
        c = kernel_semivariogram(h, 1.5, tl, tu, 2, b)["covariance"]
        for a, e in zip(c, ref):
            assert abs(a - e) < 1e-13


def test_kernel_covariance_at_long_lags():
    # 1.5 * integrate(besselJ(h w, 0), 0.1, 0.3, rel.tol = 1e-14) / 0.2 at h = 200, 450
    c = kernel_semivariogram([200, 450], 1.5, 0.1, 0.3, 2, 1.0)["covariance"]
    assert abs(c[0] - -0.00038512817440674435) < 1e-13
    assert abs(c[1] - 0.00045158119913447583) < 1e-13


def test_kernel_fit_reaches_the_optim_minimum():
    gh = [
        0.007731718001499521,
        0.032119773873861314,
        0.06548960554339905,
        0.1235270685278768,
        0.17729197449014697,
        0.2628109442161538,
        0.3308649625471822,
        0.43506276044506675,
    ]
    f = kernel_semivariogram(H, d=2, gamma_hat=gh, start=(0.0, 0.2))
    # optim (Nelder-Mead then BFGS) on the same profiled criterion (4.51)
    assert abs(f["q"] - 0.00019471369070002) < 1e-15
    assert close(f["theta_u"], 0.225600036776, 1e-5)
    assert close(f["sill_effective"], 1.795434, 1e-5)


def test_cnd_check_matches_helmert_eigen():
    co = [(0, 0), (1, 0), (0, 2), (3, 1), (2, 2.5), (4, 4)]
    D = [[math.dist(p, q) for q in co] for p in co]
    sph = [[0 if d == 0 else (1.5 * d / 3.5 - 0.5 * (d / 3.5) ** 3 if d <= 3.5 else 1.0) for d in row] for row in D]
    pw = [[d**2.5 for d in row] for row in D]
    # max(eigen(t(Q) %*% G %*% Q)$values), Q the normalised Helmert contrasts
    a = semivariogram_cnd_check(sph)
    b = semivariogram_cnd_check(pw)
    assert close(a["max_eigenvalue"], -0.39425481876645674, 1e-12) and a["valid"]
    assert close(b["max_eigenvalue"], 10.041207757544395, 1e-12) and not b["valid"]
