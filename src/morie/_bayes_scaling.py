"""Bayesian spatial-scaling samplers, the models of the R arm.

Pure-Python mirrors of rmorie's spatial_voting_bayes_native.R: Bayesian
Aldrich-McKelvey (Hare et al. 2015, the authors' JAGS model), Bakker and
Poole (2013) lognormal Bayesian MDS and unfolding, the Quinn (2004)
ordinal factor model with Cowles (1996) cutpoint steps, and
alpha-NOMINATE (Carroll et al. 2013).  Conjugate Gibbs steps where the
model admits them and slice sampling (Neal 2003, stepping out then
shrinkage) elsewhere.  The two arms draw from different generators, so
they agree in distribution (posterior means to Monte Carlo error), not
draw for draw.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from morie.fn import _array_core as np

_N01 = NormalDist()
_LOG2PI = math.log(2.0 * math.pi)


def _ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _log_ncdf(x):
    """log Phi(x), accurate in the far lower tail."""
    if x > -20.0:
        return math.log(max(0.5 * math.erfc(-x / math.sqrt(2.0)), 1e-320))
    x2 = x * x
    return -0.5 * x2 - math.log(-x) - 0.5 * _LOG2PI + math.log(1.0 - 1.0 / x2 + 3.0 / (x2 * x2))


def _qnorm(p):
    p = min(max(p, 1e-300), 1.0 - 1e-16)
    return _N01.inv_cdf(p)


def _rtnorm(rng, mu, sd, lo, hi):
    """N(mu, sd^2) truncated to [lo, hi] by inverting the CDF."""
    pl = _ncdf((lo - mu) / sd) if lo > -math.inf else 0.0
    ph = _ncdf((hi - mu) / sd) if hi < math.inf else 1.0
    if ph - pl <= 1e-300:
        return min(max(mu, lo), hi)
    x = mu + sd * _qnorm(pl + float(rng.uniform()) * (ph - pl))
    return min(max(x, lo), hi)


def _slice_vec(rng, f, x0, w, m=3, lower=-math.inf, upper=math.inf):
    """Neal (2003) stepping-out and shrinkage for independent coordinates.

    ``f`` maps a list of candidates to a list of log densities; every
    coordinate is sliced at once.
    """
    k = len(x0)
    fx = f(x0)
    y = [fx[i] - float(rng.exponential()) for i in range(k)]
    L = [x0[i] - w * float(rng.uniform()) for i in range(k)]
    R = [L[i] + w for i in range(k)]
    L = [max(v, lower) for v in L]
    R = [min(v, upper) for v in R]
    J = [math.floor(m * float(rng.uniform())) for _ in range(k)]
    K = [(m - 1) - J[i] for i in range(k)]
    while True:
        go = [J[i] > 0 and L[i] > lower for i in range(k)]
        if not any(go):
            break
        fl = f(L)
        go = [go[i] and y[i] < fl[i] for i in range(k)]
        if not any(go):
            break
        for i in range(k):
            if go[i]:
                L[i] = max(L[i] - w, lower)
                J[i] -= 1
    while True:
        go = [K[i] > 0 and R[i] < upper for i in range(k)]
        if not any(go):
            break
        fr = f(R)
        go = [go[i] and y[i] < fr[i] for i in range(k)]
        if not any(go):
            break
        for i in range(k):
            if go[i]:
                R[i] = min(R[i] + w, upper)
                K[i] -= 1
    x = list(x0)
    todo = [True] * k
    for _ in range(200):
        cand = list(x0)
        for i in range(k):
            if todo[i]:
                cand[i] = L[i] + float(rng.uniform()) * (R[i] - L[i])
        fc = f(cand)
        for i in range(k):
            if todo[i] and fc[i] > y[i]:
                x[i] = cand[i]
                todo[i] = False
            elif todo[i]:
                if cand[i] < x0[i]:
                    L[i] = cand[i]
                else:
                    R[i] = cand[i]
        if not any(todo):
            break
    return x


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    m = _mean(v)
    return math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1)) if len(v) > 1 else 0.0


def _quantile(v, p):
    """Type-7 quantile, R's default."""
    s = sorted(v)
    h = (len(s) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


# --- Bayesian Aldrich-McKelvey ------------------------------------------------


def bayes_am(Z, n_samples=1000, burn_in=200, polarity=0, thin=1, seed=42):
    """Hare et al. (2015): z_ij ~ N(a_i + b_i zhat_j, 1 / (taui_i tauj_j)).

    a_i, b_i ~ U(-100, 100); tauj ~ G(.1, .1); taui ~ G(ga, gb); ga, gb ~
    G(.1, .1); zhat the standardised zstar ~ N(0, 1), the polarity
    stimulus (0-based) truncated to the left.
    """
    rows = [[float(v) for v in r] for r in Z]
    rows = [r for r in rows if any(not math.isnan(v) for v in r)]
    N = len(rows)
    q = len(rows[0])
    if sum(any(not math.isnan(rows[i][j]) for i in range(N)) for j in range(q)) < 2:
        raise ValueError("Bayesian Aldrich-McKelvey scaling needs placements of at least two stimuli (columns).")
    if all(math.isnan(rows[i][polarity]) for i in range(N)):
        raise ValueError("The `polarity` stimulus has no placements.")
    rng = np.random.default_rng(seed)
    obs = [[not math.isnan(v) for v in r] for r in rows]
    lower = [-100.0] * q
    upper = [100.0] * q
    upper[polarity] = 0.0
    cm = []
    for j in range(q):
        v = [rows[i][j] for i in range(N) if obs[i][j]]
        cm.append(_mean(v) if v else 0.0)
    mz = _mean(cm)
    s0 = _sd(cm) or 1.0
    zs = [(t - mz) / s0 for t in cm]
    if zs[polarity] > 0:
        zs = [-t for t in zs]
    zs = [min(max(zs[j], lower[j] + 1e-8), upper[j] - 1e-8) for j in range(q)]

    def std(v):
        m = _mean(v)
        s = _sd(v)
        return [(t - m) / s for t in v]

    a = [_mean([rows[i][j] for j in range(q) if obs[i][j]]) for i in range(N)]
    b = [1.0] * N
    taui = [1.0] * N
    tauj = [1.0] * q
    ga = 1.0
    gb = 1.0
    n_i = [sum(o) for o in obs]
    n_j = [sum(obs[i][j] for i in range(N)) for j in range(q)]
    n_iter = burn_in + n_samples * thin
    keep = set(range(burn_in + thin, n_iter + 1, thin))
    out_z, out_a, out_b, out_tj = [], [], [], []
    for it in range(1, n_iter + 1):
        zh = std(zs)
        for i in range(N):
            sw = swz = sz = szz = 0.0
            for j in range(q):
                if obs[i][j]:
                    wv = taui[i] * tauj[j]
                    sw += wv
                    sz += wv * (rows[i][j] - b[i] * zh[j])
            a[i] = _rtnorm(rng, sz / sw, 1.0 / math.sqrt(sw), -100.0, 100.0)
            for j in range(q):
                if obs[i][j]:
                    wv = taui[i] * tauj[j]
                    swz += wv * zh[j] * zh[j]
                    szz += wv * zh[j] * (rows[i][j] - a[i])
            b[i] = _rtnorm(rng, szz / swz, 1.0 / math.sqrt(swz), -100.0, 100.0)
        R2 = [[(rows[i][j] - a[i] - b[i] * zh[j]) ** 2 if obs[i][j] else 0.0 for j in range(q)] for i in range(N)]
        tauj = [
            float(rng.gamma(0.1 + n_j[j] / 2.0, 1.0 / (0.1 + sum(taui[i] * R2[i][j] for i in range(N)) / 2.0)))
            for j in range(q)
        ]
        taui = [
            float(rng.gamma(ga + n_i[i] / 2.0, 1.0 / (gb + sum(R2[i][j] * tauj[j] for j in range(q)) / 2.0)))
            for i in range(N)
        ]
        gb = float(rng.gamma(0.1 + N * ga, 1.0 / (0.1 + sum(taui))))
        slt = sum(math.log(t) for t in taui)

        def f_ga(gs, gb=gb, slt=slt):
            return [
                -0.9 * math.log(g) - 0.1 * g + N * g * math.log(gb) - N * math.lgamma(g) + (g - 1.0) * slt
                if g > 0
                else -math.inf
                for g in gs
            ]

        ga = _slice_vec(rng, f_ga, [ga], 8.0, lower=0.0)[0]
        lsd = [[0.5 * math.log(taui[i] * tauj[j]) for j in range(q)] for i in range(N)]
        for j in range(q):

            def f_z(vs, j=j):
                res = []
                for vv in vs:
                    zz = list(zs)
                    zz[j] = vv
                    h = std(zz)
                    tot = -0.5 * vv * vv
                    for i in range(N):
                        for k in range(q):
                            if obs[i][k]:
                                d = rows[i][k] - a[i] - b[i] * h[k]
                                tot += lsd[i][k] - 0.5 * math.exp(2.0 * lsd[i][k]) * d * d  # noqa: B023 -- evaluated before the loop moves on
                    res.append(tot)
                return res

            zs[j] = _slice_vec(rng, f_z, [zs[j]], 8.0, lower=lower[j], upper=upper[j])[0]
        if it in keep:
            out_z.append(std(zs))
            out_a.append(list(a))
            out_b.append(list(b))
            out_tj.append(list(tauj))
    S = len(out_z)
    cols = [[out_z[s][j] for s in range(S)] for j in range(q)]
    return {
        "zeta_mean": [_mean(c) for c in cols],
        "zeta_sd": [_sd(c) for c in cols],
        "zeta_interval": [[_quantile(c, 0.025) for c in cols], [_quantile(c, 0.975) for c in cols]],
        "a": [_mean([out_a[s][i] for s in range(S)]) for i in range(N)],
        "b": [_mean([out_b[s][i] for s in range(S)]) for i in range(N)],
        "tau_stimulus": [_mean([out_tj[s][j] for s in range(S)]) for j in range(q)],
        "draws": out_z,
        "n_samples": S,
        "engine": "native Gibbs (Bayesian Aldrich-McKelvey, Hare et al. 2015)",
    }


# --- Bakker-Poole lognormal MDS and unfolding ----------------------------------


def _bp_tau(rng, n, sse):
    """tau | rest under tau ~ U(0, 10): Gamma(n/2 + 1, rate SSE/2) truncated at 10."""
    shape = n / 2.0 + 1.0
    rate = sse / 2.0
    # inverse CDF by bisection on the regularised lower gamma, which is monotone
    target = float(rng.uniform()) * _pgamma(10.0 * rate, shape)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _pgamma(mid * rate, shape) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _pgamma(x, a):
    """Regularised lower incomplete gamma P(a, x) (series / continued fraction)."""
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        tot = term
        ap = a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            tot += term
            if abs(term) < abs(tot) * 1e-16:
                break
        return tot * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _rot(X, T):
    """Orthogonal Procrustes rotation of rows X onto T (both centred)."""
    d = len(X[0])
    M = [[sum(X[i][a] * T[i][b] for i in range(len(X))) for b in range(d)] for a in range(d)]
    U, _, Vt = np.linalg.svd(np.asarray(M), full_matrices=False)
    U = [[float(U[a][k]) for k in range(d)] for a in range(d)]
    Vt = [[float(Vt[k][b]) for b in range(d)] for k in range(d)]
    return [[sum(U[a][k] * Vt[k][b] for k in range(d)) for b in range(d)] for a in range(d)]


def _rigid(X, T):
    """Translation + rotation taking X onto T; returns the map."""
    d = len(X[0])
    mx = [_mean([r[k] for r in X]) for k in range(d)]
    mt = [_mean([r[k] for r in T]) for k in range(d)]
    Q = _rot([[r[k] - mx[k] for k in range(d)] for r in X], [[r[k] - mt[k] for k in range(d)] for r in T])

    def apply(A):
        return [[sum((r[a] - mx[a]) * Q[a][b] for a in range(d)) + mt[b] for b in range(d)] for r in A]

    return apply


def _cmdscale(D, k):
    m = len(D)
    D2 = [[D[i][j] ** 2 for j in range(m)] for i in range(m)]
    rm = [_mean(r) for r in D2]
    gm = _mean(rm)
    B = [[-0.5 * (D2[i][j] - rm[i] - rm[j] + gm) for j in range(m)] for i in range(m)]
    vals, vecs = np.linalg.eigh(np.asarray(B))
    order = sorted(range(m), key=lambda t: -float(vals[t]))[:k]
    return [[float(vecs[i][t]) * math.sqrt(max(float(vals[t]), 0.0)) for t in order] for i in range(m)]


def _align_draws(draws, d, target):
    for _ in range(2):
        for s in range(len(draws)):
            draws[s] = _rigid(draws[s], target)(draws[s])
        target = [[_mean([draws[s][i][k] for s in range(len(draws))]) for k in range(d)] for i in range(len(target))]
    return draws, target


def bayes_mds(D, n_dims=2, n_samples=1000, burn_in=200, sigma_init=1.0, seed=42):
    """log delta_ij ~ N(log ||x_i - x_j||, 1/tau); x ~ N(0, 10^2); tau ~ U(0, 10)."""
    D = [[float(v) for v in r] for r in D]
    m = len(D)
    if any(len(r) != m for r in D):
        raise ValueError("`D` must be a square dissimilarity matrix.")
    ok = [
        [i != j and not math.isnan(D[min(i, j)][max(i, j)]) and D[min(i, j)][max(i, j)] > 0 for j in range(m)]
        for i in range(m)
    ]
    if not any(ok[i][j] for i in range(m) for j in range(m)):
        raise ValueError("morie_spatial_voting_bayesian_mds: D must contain positive distances.")
    LD = [[math.log(D[min(i, j)][max(i, j)]) if ok[i][j] else 0.0 for j in range(m)] for i in range(m)]
    n_obs = sum(ok[i][j] for i in range(m) for j in range(i + 1, m))
    rng = np.random.default_rng(seed)
    X = _cmdscale([[0.0 if math.isnan(v) else v for v in r] for r in D], n_dims)
    X = [[v + 1e-3 * float(rng.normal()) for v in r] for r in X]
    tau = 1.0 / sigma_init**2
    keep, ktau = [], []
    for it in range(burn_in + n_samples):
        for i in range(m):
            nb = [j for j in range(m) if ok[i][j]]
            for k in range(n_dims):
                rest = [sum((X[j][c] - X[i][c]) ** 2 for c in range(n_dims) if c != k) for j in nb]

                def f(vs, i=i, k=k, nb=nb, rest=rest):
                    out = []
                    for vv in vs:
                        t = 0.0
                        for idx, j in enumerate(nb):
                            lh = 0.5 * math.log(max(rest[idx] + (X[j][k] - vv) ** 2, 1e-24))
                            t += (LD[i][j] - lh) ** 2
                        out.append(-0.5 * tau * t - vv * vv / 200.0)  # noqa: B023 -- evaluated before the loop moves on
                    return out

                X[i][k] = _slice_vec(rng, f, [X[i][k]], 1.0)[0]
        sse = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                if ok[i][j]:
                    dd = math.sqrt(sum((X[i][c] - X[j][c]) ** 2 for c in range(n_dims)))
                    sse += (LD[i][j] - math.log(max(dd, 1e-12))) ** 2
        tau = _bp_tau(rng, n_obs, sse)
        if it >= burn_in:
            keep.append([list(r) for r in X])
            ktau.append(tau)
    keep, target = _align_draws(keep, n_dims, keep[-1])
    S = len(keep)
    dmean = [
        [
            _mean([math.sqrt(sum((keep[s][i][c] - keep[s][j][c]) ** 2 for c in range(n_dims))) for s in range(S)])
            for j in range(m)
        ]
        for i in range(m)
    ]
    return {
        "positions": target,
        "positions_sd": [[_sd([keep[s][i][k] for s in range(S)]) for k in range(n_dims)] for i in range(m)],
        "distance_mean": dmean,
        "sigma": _mean([1.0 / math.sqrt(t) for t in ktau]),
        "tau": _mean(ktau),
        "draws": keep,
        "n_samples": S,
        "engine": "native slice-within-Gibbs (Bakker-Poole Bayesian MDS)",
    }


def bayes_unfold(P, n_dims=2, n_samples=1000, burn_in=200, seed=42):
    """The same lognormal model for a respondent-by-stimulus matrix."""
    P = [[float(v) for v in r] for r in P]
    n = len(P)
    m = len(P[0])
    ok = [[not math.isnan(v) and v > 0 for v in r] for r in P]
    if not any(any(r) for r in ok):
        raise ValueError("morie_spatial_voting_bayesian_unfolding: D must contain positive dissimilarities.")
    LP = [[math.log(P[i][j]) if ok[i][j] else 0.0 for j in range(m)] for i in range(n)]
    allv = [P[i][j] for i in range(n) for j in range(m) if ok[i][j]]
    fill = _mean(allv)
    Pz = [[P[i][j] if ok[i][j] else fill for j in range(m)] for i in range(n)]
    cm = [_mean([Pz[i][j] for i in range(n)]) for j in range(m)]
    C = [[Pz[i][j] - cm[j] for j in range(m)] for i in range(n)]
    U, sv, Vt = np.linalg.svd(np.asarray(C), full_matrices=False)
    X = [[float(U[i][k]) * math.sqrt(float(sv[k])) / math.sqrt(n) for k in range(n_dims)] for i in range(n)]
    Zs = [[float(Vt[k][j]) * math.sqrt(float(sv[k])) / math.sqrt(m) for k in range(n_dims)] for j in range(m)]
    rng = np.random.default_rng(seed)
    tau = 1.0
    keepX, keepZ, ktau = [], [], []
    dsum = [[0.0] * m for _ in range(n)]

    def sq(A, B, skip=None):
        return [
            [sum((A[i][c] - B[j][c]) ** 2 for c in range(n_dims) if c != skip) for j in range(len(B))]
            for i in range(len(A))
        ]

    for it in range(burn_in + n_samples):
        for k in range(n_dims):
            base = sq(X, Zs, k)

            def fx(vs, k=k, base=base):
                return [
                    -0.5
                    * tau  # noqa: B023 -- evaluated before the loop moves on
                    * sum(
                        (LP[i][j] - 0.5 * math.log(max(base[i][j] + (vs[i] - Zs[j][k]) ** 2, 1e-24))) ** 2
                        for j in range(m)
                        if ok[i][j]
                    )
                    - vs[i] ** 2 / 200.0
                    for i in range(n)
                ]

            new = _slice_vec(rng, fx, [X[i][k] for i in range(n)], 1.0)
            for i in range(n):
                X[i][k] = new[i]
        for k in range(n_dims):
            base = sq(X, Zs, k)

            def fz(vs, k=k, base=base):
                return [
                    -0.5
                    * tau  # noqa: B023 -- evaluated before the loop moves on
                    * sum(
                        (LP[i][j] - 0.5 * math.log(max(base[i][j] + (X[i][k] - vs[j]) ** 2, 1e-24))) ** 2
                        for i in range(n)
                        if ok[i][j]
                    )
                    - vs[j] ** 2 / 200.0
                    for j in range(m)
                ]

            new = _slice_vec(rng, fz, [Zs[j][k] for j in range(m)], 1.0)
            for j in range(m):
                Zs[j][k] = new[j]
        D2 = sq(X, Zs)
        sse = sum(
            (LP[i][j] - 0.5 * math.log(max(D2[i][j], 1e-24))) ** 2 for i in range(n) for j in range(m) if ok[i][j]
        )
        tau = _bp_tau(rng, sum(sum(r) for r in ok), sse)
        if it >= burn_in:
            keepX.append([list(r) for r in X])
            keepZ.append([list(r) for r in Zs])
            ktau.append(tau)
            for i in range(n):
                for j in range(m):
                    dsum[i][j] += math.sqrt(D2[i][j])
    target = keepZ[-1]
    for _ in range(2):
        for s in range(len(keepZ)):
            f = _rigid(keepZ[s], target)
            keepX[s] = f(keepX[s])
            keepZ[s] = f(keepZ[s])
        target = [[_mean([keepZ[s][j][k] for s in range(len(keepZ))]) for k in range(n_dims)] for j in range(m)]
    S = len(keepZ)
    return {
        "stimuli": target,
        "stimuli_sd": [[_sd([keepZ[s][j][k] for s in range(S)]) for k in range(n_dims)] for j in range(m)],
        "ideal_points": [[_mean([keepX[s][i][k] for s in range(S)]) for k in range(n_dims)] for i in range(n)],
        "distance_mean": [[v / S for v in r] for r in dsum],
        "sigma": _mean([1.0 / math.sqrt(t) for t in ktau]),
        "tau": _mean(ktau),
        "n_samples": S,
        "engine": "native slice-within-Gibbs (Bakker-Poole Bayesian unfolding)",
    }


# --- Ordinal IRT (Quinn 2004) ---------------------------------------------------


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(max(s, 1e-300)) if i == j else s / L[j][j]
    return L


def _inv(A):
    n = len(A)
    M = [list(r) + [1.0 if i == j else 0.0 for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                fac = M[r][c]
                M[r] = [M[r][k] - fac * M[c][k] for k in range(2 * n)]
    return [r[n:] for r in M]


def _mvn(rng, mean, cov):
    L = _chol(cov)
    z = [float(rng.normal()) for _ in mean]
    return [mean[i] + sum(L[i][k] * z[k] for k in range(i + 1)) for i in range(len(mean))]


def ordinal_irt(Y, n_dims=1, n_samples=1000, burn_in=200, L0=0.0, seed=42):
    """y*_ij = lambda_j0 + lambda_j' phi_i + e; item cutpoints, Cowles MH."""
    raw = [[None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in r] for r in Y]
    n = len(raw)
    J = len(raw[0])
    D = int(n_dims)
    obs = [[v is not None for v in r] for r in raw]
    Yc = [[0] * J for _ in range(n)]
    ncat = []
    for j in range(J):
        lv = sorted({raw[i][j] for i in range(n) if obs[i][j]})
        if len(lv) < 2:
            raise ValueError("Every item needs at least two observed categories.")
        ncat.append(len(lv))
        idx = {v: k + 1 for k, v in enumerate(lv)}
        for i in range(n):
            if obs[i][j]:
                Yc[i][j] = idx[raw[i][j]]
    gam = [[-math.inf, 0.0] + [0.5 * (c + 1) for c in range(k - 2)] + [math.inf] for k in ncat]
    tune = [0.05 / k for k in ncat]
    acc = [0.0] * J
    rng = np.random.default_rng(seed)
    # start from the leading principal components of the mean-filled data
    cmean = [_mean([Yc[i][j] for i in range(n) if obs[i][j]]) for j in range(J)]
    M = [
        [(Yc[i][j] if obs[i][j] else cmean[j]) - cmean[j] + 1e-6 * float(rng.normal()) for j in range(J)]
        for i in range(n)
    ]
    U, sv, _ = np.linalg.svd(np.asarray(M), full_matrices=False)
    phi = [[float(U[i][k]) * float(sv[k]) for k in range(D)] for i in range(n)]
    for k in range(D):
        col = [phi[i][k] for i in range(n)]
        mc, sc = _mean(col), _sd(col) or 1.0
        for i in range(n):
            phi[i][k] = (phi[i][k] - mc) / sc
    Lam = [[0.0] * (D + 1) for _ in range(J)]
    ystar = [[0.0] * J for _ in range(n)]
    keep_phi, keep_lam = [], []
    gsum = [[0.0] * (ncat[j] - 1) for j in range(J)]
    for it in range(1, burn_in + n_samples + 1):
        mu = [[Lam[j][0] + sum(Lam[j][k + 1] * phi[i][k] for k in range(D)) for j in range(J)] for i in range(n)]
        for j in range(J):
            k = ncat[j]
            if k <= 2:
                continue
            g = gam[j]
            gp = list(g)
            for c in range(2, k):
                gp[c] = _rtnorm(rng, g[c], tune[j], gp[c - 1], g[c + 1])
            ll = 0.0
            for i in range(n):
                if obs[i][j]:
                    y = Yc[i][j]
                    m_ = mu[i][j]
                    pn = _ncdf(gp[y] - m_) - (_ncdf(gp[y - 1] - m_) if gp[y - 1] > -math.inf else 0.0)
                    po = _ncdf(g[y] - m_) - (_ncdf(g[y - 1] - m_) if g[y - 1] > -math.inf else 0.0)
                    ll += math.log(max(pn, 1e-300)) - math.log(max(po, 1e-300))
            corr = 0.0
            for c in range(2, k):
                corr += math.log(
                    max(_ncdf((g[c + 1] - g[c]) / tune[j]) - _ncdf((gp[c - 1] - g[c]) / tune[j]), 1e-300)
                ) - math.log(max(_ncdf((gp[c + 1] - gp[c]) / tune[j]) - _ncdf((g[c - 1] - gp[c]) / tune[j]), 1e-300))
            if math.log(float(rng.uniform())) < ll + corr:
                gam[j] = gp
                acc[j] += 1
        if it <= burn_in and it % 50 == 0:
            for j in range(J):
                rate = acc[j] / 50.0
                tune[j] *= 0.7 if rate < 0.2 else (1.4 if rate > 0.5 else 1.0)
                acc[j] = 0.0
        if it == burn_in:
            acc = [0.0] * J
        for i in range(n):
            for j in range(J):
                if obs[i][j]:
                    y = Yc[i][j]
                    ystar[i][j] = _rtnorm(rng, mu[i][j], 1.0, gam[j][y - 1], gam[j][y])
        for j in range(J):
            rows_ = [i for i in range(n) if obs[i][j]]
            XtX = [[0.0] * (D + 1) for _ in range(D + 1)]
            Xty = [0.0] * (D + 1)
            for i in rows_:
                xr = [1.0] + phi[i]
                for a in range(D + 1):
                    Xty[a] += xr[a] * ystar[i][j]
                    for b in range(D + 1):
                        XtX[a][b] += xr[a] * xr[b]
            for a in range(D + 1):
                XtX[a][a] += L0
            V = _inv(XtX)
            mean = [sum(V[a][b] * Xty[b] for b in range(D + 1)) for a in range(D + 1)]
            Lam[j] = _mvn(rng, mean, V)
        for i in range(n):
            js = [j for j in range(J) if obs[i][j]]
            A = [
                [(1.0 if a == b else 0.0) + sum(Lam[j][a + 1] * Lam[j][b + 1] for j in js) for b in range(D)]
                for a in range(D)
            ]
            W = _inv(A)
            rhs = [sum(Lam[j][a + 1] * (ystar[i][j] - Lam[j][0]) for j in js) for a in range(D)]
            phi[i] = _mvn(rng, [sum(W[a][b] * rhs[b] for b in range(D)) for a in range(D)], W)
        if it > burn_in:
            keep_phi.append([list(r) for r in phi])
            keep_lam.append([list(r) for r in Lam])
            for j in range(J):
                for c in range(ncat[j] - 1):
                    gsum[j][c] += gam[j][c + 1]
    S = len(keep_phi)
    first = [r for r in keep_phi[0]]
    for s in range(S):
        if D == 1:
            if keep_lam[s][0][1] < 0:
                keep_phi[s] = [[-v for v in r] for r in keep_phi[s]]
                keep_lam[s] = [[r[0]] + [-v for v in r[1:]] for r in keep_lam[s]]
        elif s > 0:
            Q = _rot(keep_phi[s], first)
            keep_phi[s] = [[sum(r[a] * Q[a][b] for a in range(D)) for b in range(D)] for r in keep_phi[s]]
            keep_lam[s] = [[r[0]] + [sum(r[a + 1] * Q[a][b] for a in range(D)) for b in range(D)] for r in keep_lam[s]]
    return {
        "ideal_points": [[_mean([keep_phi[s][i][k] for s in range(S)]) for k in range(D)] for i in range(n)],
        "ideal_sd": [[_sd([keep_phi[s][i][k] for s in range(S)]) for k in range(D)] for i in range(n)],
        "discrimination": [[_mean([keep_lam[s][j][k + 1] for s in range(S)]) for k in range(D)] for j in range(J)],
        "intercept": [_mean([keep_lam[s][j][0] for s in range(S)]) for j in range(J)],
        "cutpoints": [[v / S for v in gs] for gs in gsum],
        "acceptance": [a / S for a in acc],
        "n_samples": S,
        "engine": "native Gibbs with Cowles cutpoint steps (Quinn 2004)",
    }


# --- alpha-NOMINATE (Carroll et al. 2013) ---------------------------------------


def _anom_ll(v, dy, dn, beta, alpha):
    w2 = 0.25
    quad = -0.5 * beta * w2 * (dy - dn)
    nom = beta * (math.exp(-0.5 * w2 * dy) - math.exp(-0.5 * w2 * dn))
    u = quad + alpha * (nom - quad)
    return _log_ncdf(u if v == 1 else -u)


def _riwish(rng, df, S):
    """Inverse Wishart: the inverse of a Wishart(df, S^-1) draw (Bartlett)."""
    d = len(S)
    Sinv = _inv(S)
    L = _chol(Sinv)
    A = [[0.0] * d for _ in range(d)]
    for i in range(d):
        A[i][i] = math.sqrt(float(rng.chisquare(df - i)))
        for j in range(i):
            A[i][j] = float(rng.normal())
    LA = [[sum(L[i][k] * A[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
    Wm = [[sum(LA[i][k] * LA[j][k] for k in range(d)) for j in range(d)] for i in range(d)]
    return _inv(Wm)


def alpha_nominate(
    votes, n_dims=1, n_samples=500, burn_in=100, seed=42, thin=1, lop=0.025, minvotes=20, polarity=0, constrain=False
):
    """Bayesian alpha-NOMINATE by slice-within-Gibbs (Carroll et al. 2013)."""
    V0 = [[None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in r] for r in votes]
    if any(v not in (0.0, 1.0) for r in V0 for v in r if v is not None):
        raise ValueError("`votes` must hold 1 (yea), 0 (nay) or NA (missing).")
    d = int(n_dims)
    pol = [polarity] * d if isinstance(polarity, int) else list(polarity)
    n0 = len(V0)
    m0 = len(V0[0])
    use_votes = []
    for j in range(m0):
        cast = [V0[i][j] for i in range(n0) if V0[i][j] is not None]
        if not cast:
            continue
        y = sum(cast)
        if min(y, len(cast) - y) / len(cast) >= lop:
            use_votes.append(j)
    use_legis = [i for i in range(n0) if sum(V0[i][j] is not None for j in use_votes) >= minvotes]
    if len(use_votes) < 2 or len(use_legis) < d + 2:
        raise ValueError(
            "Too few roll calls or legislators survive the `lop` and `minvotes` screens; lower them or supply more votes."
        )
    if any(p not in use_legis for p in pol):
        raise ValueError("`polarity` must index legislators kept by the `minvotes` screen.")
    pol = [use_legis.index(p) for p in pol]
    V = [[V0[i][j] for j in use_votes] for i in use_legis]
    n, m = len(V), len(V[0])
    rng = np.random.default_rng(seed)
    X = [[float(rng.uniform(-1, 1)) for _ in range(d)] for _ in range(n)]
    for k in range(d):
        X[pol[k]][k] = abs(X[pol[k]][k])
    Yl = [[float(rng.uniform(-1, 1)) for _ in range(d)] for _ in range(m)]
    Nl = [[float(rng.uniform(-1, 1)) for _ in range(d)] for _ in range(m)]
    beta = 10.0
    alpha = 1.0 if constrain else 0.7
    n_iter = burn_in + n_samples * thin
    keep = set(range(burn_in + thin, n_iter + 1, thin))
    dX, dY, dN, dbeta, dalpha = [], [], [], [], []

    def sqd(a, b):
        return sum((a[k] - b[k]) ** 2 for k in range(d))

    def quad(P, z):
        return sum(z[a] * P[a][b] * z[b] for a in range(d) for b in range(d))

    def cross(A):
        return [[sum(r[a] * r[b] for r in A) for b in range(d)] for a in range(d)]

    for it in range(1, n_iter + 1):
        Sx = _riwish(rng, n - 1, cross(X))
        Sy = _riwish(rng, m - 1, cross(Yl))
        Sn = _riwish(rng, m - 1, cross(Nl))
        for which, loc, Sp in (("y", Yl, Sy), ("n", Nl, Sn)):
            for k in range(d):

                def f(vs, k=k, which=which, loc=loc, Sp=Sp):
                    out = []
                    for j in range(m):
                        z = list(loc[j])
                        z[k] = vs[j]
                        tot = 0.0
                        for i in range(n):
                            if V[i][j] is not None:
                                if which == "y":
                                    tot += _anom_ll(V[i][j], sqd(X[i], z), sqd(X[i], Nl[j]), beta, alpha)  # noqa: B023 -- evaluated before the loop moves on
                                else:
                                    tot += _anom_ll(V[i][j], sqd(X[i], Yl[j]), sqd(X[i], z), beta, alpha)  # noqa: B023 -- evaluated before the loop moves on
                        out.append(tot - quad(Sp, z) / 2.0)
                    return out

                new = _slice_vec(rng, f, [loc[j][k] for j in range(m)], 8.0)
                for j in range(m):
                    loc[j][k] = new[j]
        for k in range(d):

            def fx(vs, k=k):
                out = []
                for i in range(n):
                    z = list(X[i])
                    z[k] = vs[i]
                    tot = 0.0
                    for j in range(m):
                        if V[i][j] is not None:
                            tot += _anom_ll(V[i][j], sqd(z, Yl[j]), sqd(z, Nl[j]), beta, alpha)  # noqa: B023 -- evaluated before the loop moves on
                    out.append(tot - quad(Sx, z) / 2.0)  # noqa: B023 -- evaluated before the loop moves on
                return out

            new = _slice_vec(rng, fx, [X[i][k] for i in range(n)], 8.0)
            for i in range(n):
                X[i][k] = new[i]
        DY = [[sqd(X[i], Yl[j]) for j in range(m)] for i in range(n)]
        DN = [[sqd(X[i], Nl[j]) for j in range(m)] for i in range(n)]

        def total(bb, aa, DY=DY, DN=DN):
            return sum(
                _anom_ll(V[i][j], DY[i][j], DN[i][j], bb, aa) for i in range(n) for j in range(m) if V[i][j] is not None
            )

        beta = _slice_vec(rng, lambda bs: [total(b_, alpha) for b_ in bs], [beta], 8.0, lower=0.0)[0]  # noqa: B023 -- evaluated before the loop moves on
        if not constrain:
            alpha = _slice_vec(rng, lambda a_s: [total(beta, a_) for a_ in a_s], [alpha], 8.0, lower=0.0, upper=1.0)[0]  # noqa: B023 -- evaluated before the loop moves on
        if it in keep:
            dX.append([list(r) for r in X])
            dY.append([list(r) for r in Yl])
            dN.append([list(r) for r in Nl])
            dbeta.append(beta)
            dalpha.append(alpha)
    S = len(dX)
    for s in range(S):
        mu = [_mean([r[k] for r in dX[s]]) for k in range(d)]
        dX[s] = [[r[k] - mu[k] for k in range(d)] for r in dX[s]]
        dY[s] = [[r[k] - mu[k] for k in range(d)] for r in dY[s]]
        dN[s] = [[r[k] - mu[k] for k in range(d)] for r in dN[s]]
    target = dX[-1]
    for _ in range(2):
        for s in range(S):
            Q = _rot(dX[s], target)
            rot = lambda A, Q=Q: [[sum(r[a] * Q[a][b] for a in range(d)) for b in range(d)] for r in A]  # noqa: E731
            dX[s], dY[s], dN[s] = rot(dX[s]), rot(dY[s]), rot(dN[s])
        target = [[_mean([dX[s][i][k] for s in range(S)]) for k in range(d)] for i in range(n)]
    for k in range(d):
        if target[pol[k]][k] < 0:
            for s in range(S):
                for A in (dX[s], dY[s], dN[s]):
                    for r in A:
                        r[k] = -r[k]
            for r in target:
                r[k] = -r[k]
    return {
        "ideal_points": target,
        "ideal_sd": [[_sd([dX[s][i][k] for s in range(S)]) for k in range(d)] for i in range(n)],
        "yea_locations": [[_mean([dY[s][j][k] for s in range(S)]) for k in range(d)] for j in range(m)],
        "nay_locations": [[_mean([dN[s][j][k] for s in range(S)]) for k in range(d)] for j in range(m)],
        "alpha": _mean(dalpha),
        "alpha_interval": [_quantile(dalpha, 0.025), _quantile(dalpha, 0.975)],
        "beta": _mean(dbeta),
        "draws": {"X": dX, "Y": dY, "N": dN, "beta": dbeta, "alpha": dalpha},
        "legislators_used": use_legis,
        "votes_used": use_votes,
        "n_dims": d,
        "engine": "native alpha-NOMINATE slice-within-Gibbs sampler",
    }


# --- Dynamic IRT (Martin & Quinn 2002) -------------------------------------------


def dynamic_irt(
    votes,
    period,
    n_samples=500,
    burn_in=100,
    thin=1,
    seed=42,
    tau2=1.0,
    e0=0.0,
    E0=1.0,
    a0=0.0,
    A0=0.1,
    b0=0.0,
    B0=0.1,
    c0=-1.0,
    d0=-1.0,
    anchor=None,
):
    """z_jk = -alpha_k + beta_k theta_{j,t(k)} + e; theta_{j,0} ~ N(e0, E0),
    theta_{j,t} ~ N(theta_{j,t-1}, tau2_j); (alpha, beta) ~ N((a0, b0),
    diag(1/A0, 1/B0)); tau2_j ~ IG(c0/2, d0/2) when c0, d0 > 0, else fixed.
    Each sweep: truncated-normal utilities, conjugate (alpha, beta), a
    forward-filter backward-sample path per legislator, the variances.
    ``anchor`` (0-based) is reflected onto the positive side.
    """
    Y = [
        [None if (v is None or (isinstance(v, float) and math.isnan(v))) else (0.0 if v < 0 else float(v)) for v in r]
        for r in votes
    ]
    if any(v not in (0.0, 1.0) for r in Y for v in r if v is not None):
        raise ValueError("votes must be 0/1 (or -1/1) with NA for missing")
    N = len(Y)
    K = len(Y[0])
    if len(period) != K:
        raise ValueError(f"time_periods must give one period per roll call (ncol(votes) = {K}, got {len(period)})")
    levels = sorted(set(period))
    per = [levels.index(p) for p in period]
    Tn = len(levels)
    rng = np.random.default_rng(seed)
    Yc = [[0.5 if v is None else v for v in r] for r in Y]
    cm = [_mean([Yc[i][k] for i in range(N)]) for k in range(K)]
    U, sv, _ = np.linalg.svd(np.asarray([[Yc[i][k] - cm[k] for k in range(K)] for i in range(N)]), full_matrices=False)
    pc = [float(U[i][0]) * float(sv[0]) for i in range(N)]
    s = _sd(pc)
    pc = [(v - _mean(pc)) / s for v in pc] if s > 0 else [0.0] * N
    if anchor is None:
        anchor = max(range(N), key=lambda i: pc[i])
    theta = [[pc[i]] * Tn for i in range(N)]
    alpha = [0.0] * K
    beta = [1.0] * K
    t2 = [float(tau2)] * N
    est_tau = c0 > 0 and d0 > 0
    n_iter = burn_in + n_samples * thin
    keep = set(range(burn_in + thin, n_iter + 1, thin))
    S_th = [[0.0] * Tn for _ in range(N)]
    S_th2 = [[0.0] * Tn for _ in range(N)]
    S_a = [0.0] * K
    S_b = [0.0] * K
    S_t2 = [0.0] * N
    mcount = 0
    Z = [[0.0] * K for _ in range(N)]
    by_t = [[k for k in range(K) if per[k] == t] for t in range(Tn)]
    for it in range(1, n_iter + 1):
        for i in range(N):
            for k in range(K):
                mu = theta[i][per[k]] * beta[k] - alpha[k]
                v = Y[i][k]
                lo = 0.0 if v == 1.0 else -math.inf
                hi = 0.0 if v == 0.0 else math.inf
                Z[i][k] = _rtnorm(rng, mu, 1.0, lo, hi) if v is not None else mu + float(rng.normal())
        for k in range(K):
            x = [theta[i][per[k]] for i in range(N)]
            sx = sum(x)
            XtX = [[N + A0, -sx], [-sx, sum(t * t for t in x) + B0]]
            Xtz = [-sum(Z[i][k] for i in range(N)) + A0 * a0, sum(x[i] * Z[i][k] for i in range(N)) + B0 * b0]
            V = _inv(XtX)
            draw = _mvn(rng, [V[0][0] * Xtz[0] + V[0][1] * Xtz[1], V[1][0] * Xtz[0] + V[1][1] * Xtz[1]], V)
            alpha[k], beta[k] = draw[0], draw[1]
        for i in range(N):
            mf = [0.0] * Tn
            Pf = [0.0] * Tn
            m_prev, P_prev = e0, E0
            for t in range(Tn):
                prec = sum(beta[k] ** 2 for k in by_t[t])
                lin = sum((Z[i][k] + alpha[k]) * beta[k] for k in by_t[t])
                Pp = P_prev + t2[i]
                Pf[t] = 1.0 / (1.0 / Pp + prec)
                mf[t] = Pf[t] * (m_prev / Pp + lin)
                m_prev, P_prev = mf[t], Pf[t]
            theta[i][Tn - 1] = mf[Tn - 1] + math.sqrt(Pf[Tn - 1]) * float(rng.normal())
            for t in range(Tn - 2, -1, -1):
                G = Pf[t] / (Pf[t] + t2[i])
                mb = mf[t] + G * (theta[i][t + 1] - mf[t])
                theta[i][t] = mb + math.sqrt(Pf[t] * (1.0 - G)) * float(rng.normal())
        if est_tau and Tn > 1:
            for i in range(N):
                ss = sum((theta[i][t + 1] - theta[i][t]) ** 2 for t in range(Tn - 1))
                t2[i] = 1.0 / float(rng.gamma((c0 + Tn - 1) / 2.0, 2.0 / (d0 + ss)))
        if _mean(theta[anchor]) < 0:
            theta = [[-v for v in r] for r in theta]
            beta = [-v for v in beta]
        if it in keep:
            mcount += 1
            for i in range(N):
                for t in range(Tn):
                    S_th[i][t] += theta[i][t]
                    S_th2[i][t] += theta[i][t] ** 2
                S_t2[i] += t2[i]
            for k in range(K):
                S_a[k] += alpha[k]
                S_b[k] += beta[k]
    th = [[v / mcount for v in r] for r in S_th]
    return {
        "theta": th,
        "theta_sd": [[math.sqrt(max(S_th2[i][t] / mcount - th[i][t] ** 2, 0.0)) for t in range(Tn)] for i in range(N)],
        "alpha": [v / mcount for v in S_a],
        "beta": [v / mcount for v in S_b],
        "tau2": [v / mcount for v in S_t2],
        "periods": levels,
        "n_samples": mcount,
        "anchor": anchor,
        "engine": "native Gibbs sampler (Martin and Quinn 2002)",
    }
