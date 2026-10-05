"""Weak-instrument diagnostics computed from the data (the R arm's iv_weak.R).

The Kleibergen-Paap (2006) rk statistic and its Wald F (rule of thumb 10),
and the Montiel Olea-Pflueger (2013) effective F with Patnaik critical
values for the simplified, TSLS and LIML tests.  Both are robust to
heteroskedasticity and clustering (MOP also to serial correlation, HAC),
where the Stock-Yogo (2005) tables assume iid errors; nothing is looked up
in a table.
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _stats_core as stats


def _col(data, name):
    return [float(v) for v in list(data[name])]


def _t(A):
    return [list(r) for r in zip(*A)] if A else []


def _mm(A, B):
    Bt = _t(B)
    return [[sum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def _cross(A, B=None):
    """A'B (A'A when B is None) for row-major n x p matrices."""
    B = A if B is None else B
    p, q = len(A[0]), len(B[0])
    return [[sum(A[i][a] * B[i][b] for i in range(len(A))) for b in range(q)] for a in range(p)]


def _inv(A):
    n = len(A)
    M = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(2 * n)]
    return [r[n:] for r in M]


def _chol_upper(A):
    """Upper-triangular R with R'R = A (R's chol())."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    return _t(L)


def _kron(A, B):
    return [[a * b for a in ra for b in rb] for ra in A for rb in B]


def _eigh(A):
    vals, vecs = np.linalg.eigh(np.asarray(A))
    n = len(A)
    return [float(v) for v in vals], [[float(vecs[i][k]) for k in range(n)] for i in range(n)]


def _evals(A):
    S = [[(A[i][j] + A[j][i]) / 2.0 for j in range(len(A))] for i in range(len(A))]
    return _eigh(S)[0]


def _msqrt(A):
    vals, V = _eigh(A)
    n = len(A)
    return [
        [sum(V[i][k] * math.sqrt(max(vals[k], 0.0)) * V[j][k] for k in range(n)) for j in range(n)] for i in range(n)
    ]


def _gs_basis(cols):
    """Orthonormal basis of the column space (modified Gram-Schmidt, rank rule 1e-7)."""
    basis = []
    for c in cols:
        v = list(c)
        n0 = math.sqrt(sum(t * t for t in v))
        for q in basis:
            d = sum(a * b for a, b in zip(v, q))
            v = [a - d * b for a, b in zip(v, q)]
        nv = math.sqrt(sum(t * t for t in v))
        if n0 > 0 and nv > 1e-7 * n0:
            basis.append([t / nv for t in v])
    return basis


def _resid(basis, v):
    r = list(v)
    for q in basis:
        d = sum(a * b for a, b in zip(r, q))
        r = [a - d * b for a, b in zip(r, q)]
    return r


def _prepare(data, outcome, endogenous, instruments, exogenous, cluster):
    cols = list(
        dict.fromkeys(
            ([outcome] if outcome else [])
            + list(endogenous)
            + list(instruments)
            + list(exogenous or [])
            + ([cluster] if cluster else [])
        )
    )
    names = list(data.columns) if hasattr(data, "columns") else list(data.keys())
    miss = [c for c in cols if c not in names]
    if miss:
        raise ValueError("column(s) not in `data`: " + ", ".join(miss))
    raw = {c: list(data[c]) for c in cols}
    n0 = len(raw[cols[0]])

    def bad(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    keep = [i for i in range(n0) if not any(bad(raw[c][i]) for c in cols)]
    n = len(keep)
    get = {c: [raw[c][i] for i in keep] for c in cols}
    W = [[1.0] * n] + [[float(v) for v in get[c]] for c in (exogenous or [])]
    K = len(instruments)
    if K < 1:
        raise ValueError("at least one excluded instrument is needed")
    if n <= K + len(W):
        raise ValueError("fewer observations than instruments and exogenous regressors")
    bw = _gs_basis(W)

    def part(c):
        return _resid(bw, [float(v) for v in get[c]])

    return {
        "n": n,
        "L": len(W) - 1,
        "K": K,
        "y": part(outcome) if outcome else None,
        "X": [part(c) for c in endogenous],  # column lists
        "Z": [part(c) for c in instruments],
        "cl": get[cluster] if cluster else None,
    }


def _meat(S, vcov, cl=None, lag=0):
    """Long-run variance of the score rows divided by n."""
    n = len(S)
    p = len(S[0])
    if vcov == "cluster":
        groups = {}
        for i in range(n):
            g = groups.setdefault(cl[i], [0.0] * p)
            for a in range(p):
                g[a] += S[i][a]
        G = list(groups.values())
        return [[sum(g[a] * g[b] for g in G) / n for b in range(p)] for a in range(p)]
    M = [[sum(S[i][a] * S[i][b] for i in range(n)) / n for b in range(p)] for a in range(p)]
    if vcov == "hac" and lag > 0:
        for j in range(1, lag + 1):
            w = 1.0 - j / (lag + 1.0)
            Gj = [[sum(S[i][a] * S[i - j][b] for i in range(j, n)) / n for b in range(p)] for a in range(p)]
            for a in range(p):
                for b in range(p):
                    M[a][b] += w * (Gj[a][b] + Gj[b][a])
    return M


def kleibergen_paap(data, endogenous, instruments, exogenous=None, vcov="robust", cluster=None):
    """Kleibergen-Paap rk statistic and rk Wald F (ranktest's SVD form)."""
    if vcov not in ("robust", "iid", "cluster"):
        raise ValueError("vcov must be 'robust', 'iid' or 'cluster'")
    if vcov == "cluster" and not cluster:
        raise ValueError('vcov = "cluster" needs `cluster`')
    p = _prepare(data, None, endogenous, instruments, exogenous, cluster if vcov == "cluster" else None)
    n, K = p["n"], p["K"]
    k = len(p["X"])
    if k > K:
        raise ValueError("fewer excluded instruments than endogenous regressors")
    Z = _t(p["Z"])
    X = _t(p["X"])
    Qzz = [[v / n for v in r] for r in _cross(Z)]
    Qxx = [[v / n for v in r] for r in _cross(X)]
    Pi = _mm(_inv(Qzz), [[v / n for v in r] for r in _cross(Z, X)])
    ZPi = _mm(Z, Pi)
    V = [[X[i][j] - ZPi[i][j] for j in range(k)] for i in range(n)]
    Rz = _chol_upper(Qzz)
    Rx = _chol_upper(Qxx)
    Rxi = _inv(Rx)
    Theta = _mm(_mm(Rz, Pi), Rxi)
    if vcov == "iid":
        S = _kron([[v / n for v in r] for r in _cross(V)], Qzz)
    else:
        scores = [[V[i][j] * Z[i][a] for j in range(k) for a in range(K)] for i in range(n)]
        S = _meat(scores, vcov, p["cl"])
    Tm = _kron(_t(Rxi), _t(_inv(Rz)))
    Vtheta = _mm(_mm(Tm, S), _t(Tm))
    q = k - 1
    U, sv, Vt = np.linalg.svd(np.asarray(Theta), full_matrices=True)
    u = [[float(U[i][j]) for j in range(K)] for i in range(K)]
    v = [[float(Vt[j][i]) for j in range(k)] for i in range(k)]  # columns = right vectors
    iu = list(range(q, K))
    iv = list(range(q, k))
    u22 = [[u[a][b] for b in iu] for a in iu]
    v22 = [[v[a][b] for b in iv] for a in iv]
    aq = _mm(_mm([[u[a][b] for b in iu] for a in range(K)], _inv(u22)), _msqrt(_mm(u22, _t(u22))))
    bq = _mm(_mm(_msqrt(_mm(v22, _t(v22))), _inv(_t(v22))), _t([[v[a][b] for b in iv] for a in range(k)]))
    Bm = _kron(bq, _t(aq))
    vecTheta = [Theta[a][j] for j in range(k) for a in range(K)]
    lam = [sum(r[c] * vecTheta[c] for c in range(len(vecTheta))) for r in Bm]
    Vlam = _mm(_mm(Bm, Vtheta), _t(Bm))
    sol = _mm(_inv(Vlam), [[t] for t in lam])
    rk = n * sum(lam[i] * sol[i][0] for i in range(len(lam)))
    dfree = (K - q) * (k - q)
    L_all = K + p["L"] + 1
    if vcov == "cluster":
        G = len(set(p["cl"]))
        Fw = rk / (n - 1) * (n - L_all) * (G - 1) / G / K
    else:
        Fw = rk / n * (n - L_all) / K
    return {
        "statistic": Fw,
        "chi2_statistic": rk,
        "df": dfree,
        "p_value": float(stats.chi2.sf(rk, dfree)),
        "rule_of_thumb": 10.0,
        "weak": Fw < 10,
        "name": "Kleibergen-Paap rk Wald F",
        "vcov": vcov,
        "singular_values": [float(s) for s in sv],
    }


def _patnaik(W2, alpha, x):
    w = _evals(W2)
    s = sum(w)
    w = [t / s for t in w]
    K_eff = 2.0 * (1.0 + 2.0 * x) / (2.0 * sum(t * t for t in w) + 4.0 * x * max(w))
    cv = float(stats.ncx2.ppf(1.0 - alpha, K_eff, K_eff * x)) / K_eff
    return cv, K_eff


def _tr(A):
    return sum(A[i][i] for i in range(len(A)))


def _btsls_at(beta, W1, W12, W2):
    K = len(W2)
    S12 = [[W12[i][j] - beta * W2[i][j] for j in range(K)] for i in range(K)]
    S1 = [[W1[i][j] - 2 * beta * W12[i][j] + beta * beta * W2[i][j] for j in range(K)] for i in range(K)]
    ev = _evals(S12)
    t12 = _tr(S12)
    base = t12 / math.sqrt(_tr(W2) * _tr(S1))
    return max(abs(base * (1 - 2 * min(ev) / t12)), abs(base * (1 - 2 * max(ev) / t12)))


def _bliml_at(beta, W1, W12, W2, Om):
    K = len(W2)
    S12 = [[W12[i][j] - beta * W2[i][j] for j in range(K)] for i in range(K)]
    S1 = [[W1[i][j] - 2 * beta * W12[i][j] + beta * beta * W2[i][j] for j in range(K)] for i in range(K)]
    sig12 = Om[0][1] - beta * Om[1][1]
    sig1 = Om[0][0] - 2 * beta * Om[0][1] + beta * beta * Om[1][1]
    M = [[2 * S12[i][j] - sig12 / sig1 * S1[i][j] for j in range(K)] for i in range(K)]
    ev = _evals(M)
    base = _tr(S12) - sig12 / sig1 * _tr(S1)
    den = math.sqrt(_tr(W2) * _tr(S1))
    return max(abs((base - min(ev)) / den), abs((base - max(ev)) / den))


def _maximise(f, a, b, tol=1.220703e-4):
    """Golden-section maximum on [a, b] (R's optimize tolerance)."""
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol * (abs(c) + abs(d)) / 2 + 1e-12:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return max(fc, fd)


def _sup_beta(f, limit, eps=1e-3, points=10000):
    def off(b):
        return max(abs(f(b) / limit - 1), abs(f(-b) / limit - 1))

    b = 1.0
    while off(b) > eps and b < 1e8:
        b *= 2
    grid = [-b + 2 * b * i / points for i in range(points + 1)]
    vals = [f(t) for t in grid]
    i = max(range(len(vals)), key=lambda j: vals[j])
    h = 2 * b / points
    return max(vals[i], _maximise(f, grid[i] - h, grid[i] + h), limit)


def montiel_olea_pflueger(
    data,
    outcome,
    endogenous,
    instruments,
    exogenous=None,
    vcov="robust",
    cluster=None,
    lag=None,
    alpha=0.05,
    tau=(0.05, 0.10, 0.20, 0.30),
):
    """Montiel Olea-Pflueger effective F with Patnaik critical values."""
    if vcov not in ("robust", "iid", "cluster", "hac"):
        raise ValueError("vcov must be 'robust', 'iid', 'cluster' or 'hac'")
    endogenous = [endogenous] if isinstance(endogenous, str) else list(endogenous)
    if len(endogenous) != 1:
        raise ValueError(
            "the Montiel Olea-Pflueger test is for one endogenous regressor; use kleibergen_paap() with several"
        )
    if vcov == "cluster" and not cluster:
        raise ValueError('vcov = "cluster" needs `cluster`')
    p = _prepare(data, outcome, endogenous, instruments, exogenous, cluster if vcov == "cluster" else None)
    n, K, L = p["n"], p["K"], p["L"]
    dof = n - K - L - 1
    Qb = _gs_basis(p["Z"])
    Q = [[q[i] * math.sqrt(n) for q in Qb] for i in range(n)]  # n x K, Q'Q = n I
    x2 = p["X"][0]
    e1 = _resid(Qb, p["y"])
    e2 = _resid(Qb, x2)
    E = [[e1[i], e2[i]] for i in range(n)]
    Om = [[v / dof for v in r] for r in _cross(E)]
    if vcov == "iid":
        Wfull = [
            [v * n / dof for v in r]
            for r in _kron([[v / n for v in r] for r in _cross(E)], [[v / n for v in r] for r in _cross(Q)])
        ]
    else:
        if vcov != "hac":
            lag = 0
        elif lag is None:
            lag = math.floor(4 * (n / 100) ** (2 / 9))
        S = [[e1[i] * Q[i][a] for a in range(K)] + [e2[i] * Q[i][a] for a in range(K)] for i in range(n)]
        Wfull = [[v * n / dof for v in r] for r in _meat(S, vcov, p["cl"], int(lag))]
    W1 = [r[:K] for r in Wfull[:K]]
    W12 = [r[K:] for r in Wfull[:K]]
    W2 = [r[K:] for r in Wfull[K:]]
    if vcov == "cluster":
        G = len(set(p["cl"]))
        adj = (n / (n - 1)) * (G - 1) / G
    else:
        adj = 1.0
    qx = [sum(Q[i][a] * x2[i] for i in range(n)) for a in range(K)]
    F_eff = adj * sum(t * t for t in qx) / n / _tr(W2)
    ev2 = _evals(W2)
    B_tsls = _sup_beta(lambda b: _btsls_at(b, W1, W12, W2), 1 - 2 * min(ev2) / sum(ev2))
    B_liml = _sup_beta(lambda b: _bliml_at(b, W1, W12, W2, Om), max(ev2) / sum(ev2))
    rows = []
    for t in tau:
        s = _patnaik(W2, alpha, 1 / t)
        g = _patnaik(W2, alpha, B_tsls / t)
        lm = _patnaik(W2, alpha, B_liml / t)
        rows.append(
            {
                "tau": t,
                "simplified": s[0],
                "tsls": g[0],
                "liml": lm[0],
                "K_eff_simplified": s[1],
                "K_eff_tsls": g[1],
                "K_eff_liml": lm[1],
            }
        )
    return {
        "F_eff": F_eff,
        "critical_values": rows,
        "B_tsls": B_tsls,
        "B_liml": B_liml,
        "reject": {f"tau_{round(100 * r['tau'])}": F_eff > r["tsls"] for r in rows},
        "n": n,
        "K": K,
        "vcov": vcov,
        "alpha": alpha,
        "name": "Montiel Olea-Pflueger effective F",
    }
