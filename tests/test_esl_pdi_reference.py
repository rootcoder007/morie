"""ProDenICA (ESL Algorithm 14.3, eqs 14.89-14.97): recovery, whitening, Poisson score equations, fixed point, R parity."""

import math

from morie.fn.eslpdi import _fit_tilt, _inv_sqrt, _matmul, _spline_eval, esl_prodenica


def mixed():
    xs, S = 12345, []
    for _ in range(400):
        u = []
        for _ in range(2):
            xs = xs * 16807 % 2147483647  # Park-Miller, exact in doubles
            u.append(xs / 2147483647)
        S.append([(u[0] - 0.5) * 12**0.5, -math.log(u[1]) - 1])  # uniform and exponential sources
    return [[a + 0.6 * b, 0.4 * a + b] for a, b in S]  # X = S M, M = [[1, .4], [.6, 1]]


def test_recovery_whitening_and_parity():
    X = mixed()
    r = esl_prodenica(X)
    U, A, S = r["unmixing"], r["A"], r["sources"]
    M = [[1, 0.4], [0.6, 1]]
    R = [[sum(M[i][k] * U[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    amari = (
        sum(sum(abs(v) for v in row) / max(abs(v) for v in row) - 1 for row in R)
        + sum(sum(abs(R[i][j]) for i in range(2)) / max(abs(R[i][j]) for i in range(2)) - 1 for j in range(2))
    ) / 4  # (14.97)
    assert r["converged"] and amari < 0.02  # CRAN ProDenICA reaches 0.0121 on these data
    for i in range(2):
        for j in range(2):
            assert abs(sum(A[c][i] * A[c][j] for c in range(2)) - (i == j)) < 1e-12
            assert abs(sum(s[i] * s[j] for s in S) / 400 - (i == j)) < 1e-10  # whitened sources
    mean = r["mean"]
    assert (
        max(abs(s[j] - sum((x[c] - mean[c]) * U[c][j] for c in range(2))) for s, x in zip(S, X) for j in range(2))
        < 1e-12
    )
    assert abs(r["negentropy"] - 0.51945264503196764) < 1e-9  # R arm


def test_tilt_fit_matches_the_poisson_score_equations():
    s = [row[0] for row in esl_prodenica(mixed(), max_iter=1)["sources"]]
    u, f, g = _fit_tilt(s, 500, 1e-4)
    d = u[1] - u[0]
    ys = [0.0] * 500
    for v in s:
        ys[int((v - u[0] + d / 2) / d)] += 1 / 400
    mu = [d * math.exp(-t * t / 2) / math.sqrt(2 * math.pi) * math.exp(v) for t, v in zip(u, f)]
    # unpenalised constant and linear terms: fitted mass and first moment equal the binned data's
    assert abs(sum(mu) - 1) < 1e-8
    assert abs(sum(m * t for m, t in zip(mu, u)) - sum(y * t for y, t in zip(ys, u))) < 1e-8
    k = 250
    e = 1e-6
    t = u[k] + 0.3 * d
    val, d1, d2 = _spline_eval(u, f, g, t)
    assert abs((_spline_eval(u, f, g, t + e)[0] - _spline_eval(u, f, g, t - e)[0]) / (2 * e) - d1) < 1e-6
    assert abs((_spline_eval(u, f, g, t + e)[1] - _spline_eval(u, f, g, t - e)[1]) / (2 * e) - d2) < 1e-5


def test_fixed_point_step():
    X = mixed()
    r = esl_prodenica(X)
    A, Z = r["A"], [[sum((x[c] - r["mean"][c]) * r["whitening"][c][k] for c in range(2)) for k in range(2)] for x in X]
    cols = []
    for j in range(2):
        s = [sum(z[c] * A[c][j] for c in range(2)) for z in Z]
        u, f, g = _fit_tilt(s, 500, 1e-4)
        ev = [_spline_eval(u, f, g, v) for v in s]
        m2 = sum(e[2] for e in ev) / 400
        cols.append([sum(z[c] * e[1] for z, e in zip(Z, ev)) / 400 - m2 * A[c][j] for c in range(2)])  # (14.96)
    An = [list(r) for r in zip(*cols)]
    An = _matmul(An, _inv_sqrt(_matmul([list(r) for r in zip(*An)], An)))  # A <- U V^T
    assert max(1 - abs(sum(An[c][j] * A[c][j] for c in range(2))) for j in range(2)) < 1e-8  # one more step stays put
