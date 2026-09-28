"""sfilter: eigenvector maps checked against their defining properties; selection recomputed by brute force."""

import math

import pytest

from morie.fn.sfilter import eigenvector_filtering, getis_filter, moran_eigenvectors

PTS = [(0.0, 0.0), (1.0, 0.2), (2.1, 0.0), (0.1, 1.3), (1.2, 1.1), (2.0, 1.4), (0.6, 2.4), (1.7, 2.6)]
W = [[1.0 if 0 < math.dist(a, b) < 1.6 else 0.0 for b in PTS] for a in PTS]


def test_eigenvectors_properties():
    r = moran_eigenvectors(W)
    n = 8
    V, lam = r.vectors, r.eigenvalues
    assert len(lam) == n - 1 and lam == sorted(lam, reverse=True)
    s0 = sum(map(sum, W))
    M = [[(1.0 if i == j else 0.0) - 1 / n for j in range(n)] for i in range(n)]
    MWM = [[sum(M[i][a] * W[a][b] * M[b][j] for a in range(n) for b in range(n)) for j in range(n)] for i in range(n)]
    for k in range(n - 1):
        v = [V[i][k] for i in range(n)]
        assert sum(v) == pytest.approx(0.0, abs=1e-12) and sum(x * x for x in v) == pytest.approx(1.0)
        Av = [sum(MWM[i][j] * v[j] for j in range(n)) for i in range(n)]
        assert Av == pytest.approx([lam[k] * x for x in v], abs=1e-12)
        vWv = sum(v[i] * W[i][j] * v[j] for i in range(n) for j in range(n))
        assert r.moran_i[k] == pytest.approx(n / s0 * vWv, abs=1e-12)
        assert next(x for x in v if abs(x) > 1e-12) > 0
    assert sorted(r.positive + r.negative) == list(range(n - 1))


def test_selection_matches_brute_force_first_step():
    y = [1.0, 1.4, 2.2, 1.6, 2.0, 2.9, 2.4, 3.3]
    me = moran_eigenvectors(W)
    r = eigenvector_filtering(y, W, criterion="r2", tol=0.0, max_vectors=1)

    def r2(k):
        X = [[1.0, me.vectors[i][k]] for i in range(8)]
        # the eigenvectors are orthogonal to the constant, so the OLS slope is v'y
        b = sum(X[i][1] * y[i] for i in range(8))
        m = sum(y) / 8
        return 1 - sum((y[i] - m - b * X[i][1]) ** 2 for i in range(8)) / sum((v - m) ** 2 for v in y)

    best = max(me.positive, key=lambda k: (r2(k), -k))
    assert r.selected == [best] and r.r2 == pytest.approx(r2(best))

    def ols(sel):
        X = [[1.0] + [me.vectors[i][k] for k in sel] for i in range(8)]
        p = len(X[0])
        XtX = [[sum(r[u] * r[v] for r in X) for v in range(p)] for u in range(p)]
        Xty = [sum(r[u] * yy for r, yy in zip(X, y)) for u in range(p)]
        # Gauss-Jordan solve
        A = [row[:] + [t] for row, t in zip(XtX, Xty)]
        for c in range(p):
            piv = A[c][c]
            A[c] = [v / piv for v in A[c]]
            for rr in range(p):
                if rr != c:
                    f = A[rr][c]
                    A[rr] = [v - f * w for v, w in zip(A[rr], A[c])]
        beta = [A[u][p] for u in range(p)]
        res = [yy - sum(bb * v for bb, v in zip(beta, r)) for r, yy in zip(X, y)]
        rss = sum(e * e for e in res)
        return n * math.log(rss / n) + 2 * (p + 1), res

    n = 8
    a = eigenvector_filtering(y, W, criterion="aic")
    aic_sel, res_sel = ols(a.selected)
    assert a.aic == pytest.approx(aic_sel)
    # stopping rule: no remaining positive eigenvector lowers the AIC further
    assert all(ols(a.selected + [k])[0] >= a.aic - 1e-12 for k in me.positive if k not in a.selected)
    m = eigenvector_filtering(y, W, criterion="moran", tol=0.05)
    _, res_m = ols(m.selected)
    d = [e - sum(res_m) / 8 for e in res_m]
    mi = 8 / sum(map(sum, W)) * sum(W[i][j] * d[i] * d[j] for i in range(8) for j in range(8)) / sum(v * v for v in d)
    assert m.residual_moran == pytest.approx(mi, abs=1e-12)
    # the residual autocorrelation is negative, so positive filters cannot bring |I| below tol: all are used
    assert abs(m.residual_moran) >= 0.05 and sorted(m.selected) == sorted(me.positive)
    assert m.residuals == pytest.approx(res_m, abs=1e-12)


def test_getis_filter():
    x = [3.0, 5.0, 2.0, 8.0, 4.0, 6.0, 1.0, 7.0]
    g = getis_filter(x, W)
    tot = sum(x)
    for i in range(8):
        wi = sum(W[i][j] for j in range(8) if j != i)
        gi = sum(W[i][j] * x[j] for j in range(8) if j != i) / (tot - x[i])
        assert g.filtered[i] == pytest.approx(x[i] * (wi / 7) / gi)
        assert g.spatial[i] == pytest.approx(x[i] - g.filtered[i])
