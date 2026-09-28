import math

from morie.fn._qpcore import ssum
from morie.fn.smallarea import bhf_eblup, fay_herriot, marshall_eb, potthoff_whittinghill

FY = [10.2, 12.8, 11.3, 15.0, 13.2, 9.7, 16.5, 8.9, 14.1, 12.0]
FX = [3.1, 2.2, 4.0, 2.9, 4.4, 1.9, 3.6, 2.7, 4.8, 3.3]
FD = [1.2, 0.8, 2.0, 1.5, 0.9, 1.1, 1.7, 0.6, 1.3, 1.0]
X = [[1.0, x] for x in FX]


def _gls(A):
    w = [1 / (A + d) for d in FD]
    s0, s1, s2 = ssum(w), ssum(wi * x for wi, x in zip(w, FX)), ssum(wi * x * x for wi, x in zip(w, FX))
    t0, t1 = ssum(wi * y for wi, y in zip(w, FY)), ssum(wi * x * y for wi, x, y in zip(w, FX, FY))
    det = s0 * s2 - s1 * s1
    b = [(s2 * t0 - s1 * t1) / det, (s0 * t1 - s1 * t0) / det]
    return w, b, [[s2 / det, -s1 / det], [-s1 / det, s0 / det]]


def test_fay_herriot_estimating_equations_and_mse():
    for method in ("REML", "ML", "FH"):
        r = fay_herriot(FY, X, FD, method=method, tol=1e-13, maxiter=1000)
        w, b, Q = _gls(r.A)
        assert max(abs(u - v) for u, v in zip(b, r.beta)) < 1e-10
        res = [y - b[0] - b[1] * x for y, x in zip(FY, FX)]
        if method == "ML":
            assert abs(-0.5 * ssum(w) + 0.5 * ssum((wi * e) ** 2 for wi, e in zip(w, res))) < 1e-9
        if method == "FH":
            assert abs(ssum(e * e * wi for e, wi in zip(res, w)) - 8) < 1e-9
        for i in range(10):
            g = r.A / (r.A + FD[i])
            assert abs(r.eblup[i] - (b[0] + b[1] * FX[i] + g * res[i])) < 1e-10
        if method == "REML":
            B = [d * wi for d, wi in zip(FD, w)]
            varA = 2 / ssum(wi * wi for wi in w)
            for i in range(10):
                g2 = B[i] ** 2 * (Q[0][0] + 2 * Q[0][1] * FX[i] + Q[1][1] * FX[i] ** 2)
                mse = FD[i] * (1 - B[i]) + g2 + 2 * B[i] ** 2 * varA / (r.A + FD[i])
                assert abs(r.mse[i] - mse) < 1e-10
    e = fay_herriot([1.0, 2.0, 3.0, 5.0], [[1]] * 4, [1.0] * 4)
    assert abs(e.A - (8.75 / 3 - 1)) < 1e-12
    e = fay_herriot([1.0, 2.0, 3.0, 5.0], [[1]] * 4, [1.0] * 4, method="ML")
    assert abs(e.A - (8.75 / 4 - 1)) < 1e-12


AREA = [1] * 3 + [2] * 4 + [3] * 2 + [4] * 5 + [5] * 3
UX = [1.2, 2.3, 1.8, 3.1, 2.2, 2.9, 3.5, 0.8, 1.1, 2.6, 3.3, 2.8, 3.9, 3.0, 1.9, 2.4, 2.1]
UY = [4.1, 5.9, 5.0, 7.9, 5.6, 7.5, 8.9, 2.4, 3.6, 7.0, 8.1, 7.4, 9.6, 7.7, 4.4, 5.5, 5.3]


def test_bhf_balanced_anova_and_eblup_formula():
    y = [1.0, 2.0, 2.0, 3.0, 5.0, 6.0]
    r = bhf_eblup(y, [[1]] * 6, [0, 0, 1, 1, 2, 2], [[1], [1], [1]])
    means = [1.5, 2.5, 5.5]
    msb = 2 * ssum((m - 19 / 6) ** 2 for m in means) / 2
    assert abs(r.sigma2_e - 0.5) < 1e-12 and abs(r.sigma2_u - (msb - 0.5) / 2) < 1e-10
    xb = [1.7, 2.9, 1.0, 3.1, 2.2, 2.5]
    N = [50, 80, 30, 120, 60, 40]
    r = bhf_eblup(UY, [[1.0, x] for x in UX], AREA, [[1.0, v] for v in xb], areas=[1, 2, 3, 4, 5, 6], popsize=N)
    b = r.beta
    for k, lab in enumerate([1, 2, 3, 4, 5]):
        idx = [i for i, a in enumerate(AREA) if a == lab]
        n = len(idx)
        yb = ssum(UY[i] for i in idx) / n
        xs = ssum(UX[i] for i in idx) / n
        u = r.sigma2_u / (r.sigma2_u + r.sigma2_e / n) * (yb - b[0] - b[1] * xs)
        f = n / N[k]
        assert abs(r.eblup[k] - (f * yb + (1 - f) * b[0] + (xb[k] - f * xs) * b[1] + (1 - f) * u)) < 1e-12
    assert abs(r.eblup[5] - (b[0] + 2.5 * b[1])) < 1e-12 and r.sample_sizes[5] == 0
    ml = bhf_eblup(UY, [[1.0, x] for x in UX], AREA, [[1.0, v] for v in xb[:5]], method="ML")
    assert ml.sigma2_u < r.sigma2_u


def test_marshall_eb_formulas():
    n = [3, 10, 4, 25, 7, 1]
    x = [800, 1500, 1000, 3000, 1200, 400]
    r = marshall_eb(n, x)
    b = 50 / 7900
    s2 = ssum(xi * (ni / xi - b) ** 2 for ni, xi in zip(n, x)) / 7900
    a = max(0.0, s2 - b / (7900 / 6))
    assert all(abs(e - (b + a * (ni / xi - b) / (a + b / xi))) < 1e-15 for e, ni, xi in zip(r.estimate, n, x))
    r = marshall_eb([2, 8], [100, 100])
    assert [round(v, 10) for v in r.estimate] == [0.0366666667, 0.0633333333]
    rb = marshall_eb(n, x, family="binomial")
    for e, ni, xi in zip(rb.estimate, n, x):
        rho = (xi * s2 - (xi / (7900 / 6)) * b * (1 - b)) / (
            (xi - 1) * s2 + ((7900 / 6 - xi) / (7900 / 6)) * b * (1 - b)
        )
        assert abs(e - (rho * ni / xi + (1 - rho) * b)) < 1e-15


def test_potthoff_whittinghill():
    r = potthoff_whittinghill([3, 3, 3, 3], [3.0, 3.0, 3.0, 3.0])
    assert (r.T, r.mean, r.variance) == (96.0, 132.0, 792.0)
    assert abs(r.z - (96 - 132) / math.sqrt(792)) < 1e-15
    assert abs(r.p_value - min(r.p_upper, 1 - r.p_upper)) < 1e-15
