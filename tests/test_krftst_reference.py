"""Kenward-Roger F test (eqs 6.53-6.54) against pbkrtest and finite differences."""

import math

from morie.fn.krftst import kenward_roger_test


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_variance_components_match_krmodcomp():
    n = 30
    g = [i // 5 for i in range(n)]
    x = [math.sin(1.3 * i) + 0.1 * i for i in range(n)]
    eff = [-0.6, 0.3, 0.9, -0.2, 0.4, -0.8]
    y = [2 + 0.5 * x[i] + eff[g[i]] + 0.5 * math.sin(2.9 * i + 0.4) for i in range(n)]
    zz = [[1.0 if g[a] == g[b] else 0.0 for b in range(n)] for a in range(n)]
    eye = [[1.0 if a == b else 0.0 for b in range(n)] for a in range(n)]
    s = [[0.41778383883297898 * zz[a][b] + 0.13995228058289766 * eye[a][b] for b in range(n)] for a in range(n)]
    r = kenward_roger_test([[1.0, v] for v in x], y, s, [zz, eye], [[0.0, 1.0]])
    # KRmodcomp(lmer(y ~ x + (1 | g)), update(., . ~ . - x))
    assert close(r["F"], 18.458409881288013, 1e-11)
    assert close(r["ddf"], 27.367979299962421, 1e-11)
    assert r["scaling"] == 1.0
    assert close(r["p_value"], 0.00019657909126163887, 1e-10)


def _spatial():
    ns = 25
    co = [(10 * ((j * 0.618034) % 1), 10 * ((j * 0.414214 + 0.3) % 1)) for j in range(ns)]
    d = [[math.dist(p, q) for q in co] for p in co]
    xs = [[1.0, p[0]] for p in co]
    ys = [1 + 0.2 * co[j][0] + math.sin(co[j][1]) + 0.3 * math.cos(5.3 * j) for j in range(ns)]
    r = [[math.exp(-3 * v / 4) for v in row] for row in d]
    s = [[1.8 if a == b else 1.5 * r[a][b] for b in range(ns)] for a in range(ns)]
    dr = [[r[a][b] * 3 * d[a][b] / 16 for b in range(ns)] for a in range(ns)]
    d2r = [[r[a][b] * (9 * d[a][b] ** 2 / 256 - 6 * d[a][b] / 64) for b in range(ns)] for a in range(ns)]
    eye = [[1.0 if a == b else 0.0 for b in range(ns)] for a in range(ns)]
    return xs, ys, s, [eye, r, [[1.5 * v for v in row] for row in dr]], dr, d2r


def test_spatial_matches_pbkrtest_internals():
    xs, ys, s, g, _, _ = _spatial()
    r = kenward_roger_test(xs, ys, s, g, [[0.0, 1.0]])
    # pbkrtest:::vcovAdj_internal then pbkrtest:::.KR_adjust with the same Sigma and derivatives
    assert close(r["F"], 1.2806152728363422, 1e-11)
    assert close(r["ddf"], 2.0926745247882712, 1e-11)
    assert close(r["Phi_adjusted"][1][1], 0.017622414876145718, 1e-11)


def test_second_order_terms_match_finite_differences():
    xs, ys, s, g, dr, d2r = _spatial()
    r = kenward_roger_test(
        xs, ys, s, g, [[0.0, 1.0]], d2Sigma={(1, 2): dr, (2, 2): [[1.5 * v for v in row] for row in d2r]}
    )
    # Phi_A with R_ij from central differences of Sigma(theta), step 1e-4
    assert close(r["Phi_adjusted"][1][1], 0.039393056880032834, 1e-7)
    assert close(r["Phi_adjusted"][0][1], -0.20150629908819473, 1e-7)
