"""Tests for morie.fn.sdmvar: every expected value is recomputed from the formula."""

from morie.fn.sdmvar import sdmvar


def _inv(A):
    n = len(A)
    M = [list(map(float, r)) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]


def _mm(A, B):
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


def _mv(A, v):
    return [sum(a * b for a, b in zip(r, v)) for r in A]


def _t(A):
    return [list(c) for c in zip(*A)]


def _imw(W, a):
    return [[(1.0 if i == j else 0.0) - a * W[i][j] for j in range(len(W))] for i in range(len(W))]


def _det(A):
    # Laplace expansion (small matrices only)
    if len(A) == 1:
        return A[0][0]
    return sum((-1) ** j * A[0][j] * _det([r[:j] + r[j + 1 :] for r in A[1:]]) for j in range(len(A)))


N = 8
_B = [[1.0 if abs(i - j) == 1 or {i, j} == {0, 7} or {i, j} == {2, 5} else 0.0 for j in range(N)] for i in range(N)]
W = [[v / sum(r) for v in r] for r in _B]
X = [[1.0, ((i * 7) % 11) / 5] for i in range(N)]
Y = [1 + 2 * X[i][1] + ((i * 3) % 5 - 2) / 4 + 0.3 * sum(W[i][j] * X[j][1] for j in range(N)) for i in range(N)]
E = [((i * 4) % 7 - 3) / 5 for i in range(N)]
W3 = [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]


def _fisher(Xm, Wm, beta, rho, lam, s2, lag, err):
    """I_ij = dmu_i' S^-1 dmu_j + tr(S^-1 dS_i S^-1 dS_j) / 2 for y ~ N(A^-1 X b, s2 A^-1 B^-1 B^-T A^-T)."""
    n, p = len(Xm), len(Xm[0])
    A, Bm = _imw(Wm, rho), _imw(Wm, lam)
    Ai, Bi = _inv(A), _inv(Bm)
    S = _mm(Ai, Bi)
    Sig = [[s2 * v for v in r] for r in _mm(S, _t(S))]
    Sigi = _inv(Sig)
    mu_d, sig_d = [], []
    for k in range(p):
        mu_d.append(_mv(Ai, [r[k] for r in Xm]))
        sig_d.append([[0.0] * n for _ in range(n)])
    Xb = _mv(Xm, beta) if lag else [0.0] * n

    def dsig(dS):
        P = _mm(dS, _t(S))
        return [[s2 * (P[i][j] + P[j][i]) for j in range(n)] for i in range(n)]

    if lag:
        AiWAi = _mm(_mm(Ai, Wm), Ai)
        mu_d.append(_mv(AiWAi, Xb))
        sig_d.append(dsig(_mm(AiWAi, Bi)))
    if err:
        mu_d.append([0.0] * n)
        sig_d.append(dsig(_mm(_mm(S, Wm), Bi)))
    mu_d.append([0.0] * n)
    sig_d.append(_mm(S, _t(S)))
    k = len(mu_d)
    out = [[0.0] * k for _ in range(k)]
    for a in range(k):
        for b in range(k):
            m = sum(u * v for u, v in zip(mu_d[a], _mv(Sigi, mu_d[b])))
            P, Q = _mm(Sigi, sig_d[a]), _mm(Sigi, sig_d[b])
            out[a][b] = m + 0.5 * sum(P[i][j] * Q[j][i] for i in range(n) for j in range(n))
    return out


def _check(r, F):
    k = len(F)
    for a in range(k):
        for b in range(k):
            assert abs(r.extra["information"][a][b] - F[a][b]) < 1e-8 * max(1.0, abs(F[a][b]))
    ident = _mm(r.extra["cov"], F)
    for a in range(k):
        for b in range(k):
            assert abs(ident[a][b] - (1.0 if a == b else 0.0)) < 1e-8


def test_inverse_fisher_information_of_the_durbin_design():
    WX = [[sum(W[i][j] * X[j][1] for j in range(N))] for i in range(N)]
    Z = [a + b for a, b in zip(X, WX)]
    r = sdmvar(X, WX, W, 0.3, 0.5, [1.0, 2.0, 0.4])
    _check(r, _fisher(Z, W, [1.0, 2.0, 0.4], 0.3, 0.0, 0.5, True, False))
    assert r.statistic == r.extra["cov"][3][3]
