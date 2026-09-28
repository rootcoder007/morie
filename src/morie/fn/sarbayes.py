# morie.fn -- function file (rootcoder007/morie)
"""Bayesian spatial autoregressive models by Gibbs sampling (LeSage and Pace 2009): the SAR
probit, SAR Tobit and SAR ordered probit with latent variables drawn from their truncated
multivariate normal full conditionals, and the continuous spatial lag, spatial error and spatial
Durbin models. The spatial parameter is drawn by griddy Gibbs on a grid of step 0.001 with exact
log-determinants; all randomness comes from the Philox stream (``seed``)."""

from __future__ import annotations

import math

from . import _array_core as np
from ._qpcore import inverse, solve, ssum
from ._richresult import RichResult
from ._rng import random_uniform
from ._rrng_core import pnorm, qnorm

__all__ = ["sar_probit_gibbs", "sar_tobit_gibbs", "sar_ordered_probit_gibbs", "spatial_bayes_gibbs"]


def _vec(x):
    return [float(v) for v in np.asarray(x, dtype=float).ravel().tolist()]


def _mat(X):
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return [[float(v) for v in r] for r in a.tolist()]


def _mv(A, v):
    return [ssum(r[j] * v[j] for j in range(len(v))) for r in A]


def _chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - ssum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = math.sqrt(s) if i == j else s / L[j][j]
    return L


def _logdet_lu(A):
    n = len(A)
    M = [list(r) for r in A]
    ld = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if p != c:
            M[c], M[p] = M[p], M[c]
        if M[c][c] == 0.0:
            return -math.inf
        ld += math.log(abs(M[c][c]))
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f != 0.0:
                for k in range(c, n):
                    M[r][k] -= f * M[c][k]
    return ld


def _logdet_grid(W, grid):
    n = len(W)
    # row-standardised symmetric-pattern W = D^{-1} A is similar to D^{1/2} W D^{-1/2}, which is symmetric
    d = [float(sum(1 for v in r if v != 0.0)) for r in W]
    sym = all(abs(W[i][j] * d[i] - W[j][i] * d[j]) <= 1e-12 for i in range(n) for j in range(n)) and min(d) > 0
    if sym:
        M = [[W[i][j] * math.sqrt(d[i] / d[j]) for j in range(n)] for i in range(n)]
        lam = [float(v) for v in np.linalg.eigvalsh(np.asarray(M, dtype=float)).tolist()]
        return [ssum(math.log(1.0 - r * v) for v in lam) for r in grid]
    coarse = [-0.99 + 0.01 * i for i in range(199)]
    lc = [_logdet_lu([[(1.0 if i == j else 0.0) - r * W[i][j] for j in range(n)] for i in range(n)]) for r in coarse]
    out = []
    for r in grid:
        t = min(max((r + 0.99) / 0.01, 0.0), 197.999999)
        i = int(t)
        out.append(lc[i] + (t - i) * (lc[i + 1] - lc[i]))
    return out


class _Stream:
    def __init__(self, seed, n):
        self.u = [float(v) for v in random_uniform(n, seed=seed)]
        self.k = 0

    def unif(self):
        v = self.u[self.k]
        self.k += 1
        return v

    def norm(self):
        return qnorm(self.unif())


def _tnorm(m, s, lo, hi, u):
    # inverse-cdf draw from N(m, s^2) truncated to (lo, hi), tail-stable
    a = (lo - m) / s if lo > -math.inf else -math.inf
    b = (hi - m) / s if hi < math.inf else math.inf
    if b == math.inf:
        pa = pnorm(-a) if a > -math.inf else 1.0
        return m - s * qnorm(u * pa) if a > -math.inf else m + s * qnorm(u)
    if a == -math.inf:
        return m + s * qnorm(u * pnorm(b))
    fa, fb = pnorm(a), pnorm(b)
    if fb - fa > 1e-12:
        return m + s * qnorm(fa + u * (fb - fa))
    return m + s * (a + u * (b - a))


def _draw_rho(grid, lndet, lnprior, c0, c1, c2, power, u):
    # power > 0: |A| q(r)^(-power) (sigma integrated out); power == 0: |A| exp(-q(r) / 2) (sigma = 1)
    if power > 0:
        den = [
            ld + lp - power * math.log(max(c0 - 2 * r * c1 + r * r * c2, 1e-300))
            for r, ld, lp in zip(grid, lndet, lnprior)
        ]
    else:
        den = [ld + lp - 0.5 * (c0 - 2 * r * c1 + r * r * c2) for r, ld, lp in zip(grid, lndet, lnprior)]
    mx = max(den)
    w = [math.exp(v - mx) for v in den]
    tot = ssum(w)
    target = u * tot
    acc = 0.0
    for r, v in zip(grid, w):
        acc += v
        if acc >= target:
            return r
    return grid[-1]


def _setup(W, a1, a2):
    grid = [-0.999 + 0.001 * i for i in range(1999)]
    lndet = _logdet_grid(W, grid)
    lb = math.lgamma(a1 + a2) - math.lgamma(a1) - math.lgamma(a2) - (a1 + a2 - 1) * math.log(2.0)
    lnprior = [lb + (a1 - 1) * math.log(1 + r) + (a2 - 1) * math.log(1 - r) for r in grid]
    return grid, lndet, lnprior


def _latent_gibbs(
    y,
    X,
    W,
    ndraw,
    burn_in,
    seed,
    a1,
    a2,
    bounds_fn,
    sigma_fixed,
    cut_fn=None,
    observed=None,
    prior_var=None,
    exact=False,
):
    n, k = len(X), len(X[0])
    grid, lndet, lnprior = _setup(W, a1, a2)
    pp = 0.0 if prior_var is None else 1.0 / prior_var
    xtx = [[ssum(r[a] * r[b] for r in X) + (pp if a == b else 0.0) for b in range(k)] for a in range(k)]
    xtxi = inverse(xtx)
    Lb = _chol(xtxi)
    xtxi0 = inverse([[ssum(r[a] * r[b] for r in X) for b in range(k)] for a in range(k)])
    total = ndraw + burn_in
    st = _Stream(seed, total * (3 * n + k + 2))
    latent = any(bounds_fn(i)[0] is not None for i in range(n))
    WW = [[W[i][j] + W[j][i] for j in range(n)] for i in range(n)]
    WtW = [[ssum(W[m][i] * W[m][j] for m in range(n)) for j in range(n)] for i in range(n)]
    rho, beta, s2 = 0.0, [0.0] * k, 1.0
    z = [0.0] * n
    if observed is not None:
        z = [observed[i] if observed[i] is not None else 0.0 for i in range(n)]
    keep_b, keep_r, keep_s, keep_c = [], [], [], []
    for it in range(total):
        A = [[(1.0 if i == j else 0.0) - rho * W[i][j] for j in range(n)] for i in range(n)]
        if latent:
            mu = solve(A, _mv(X, beta))
            H = [
                [((1.0 if i == j else 0.0) - rho * WW[i][j] + rho * rho * WtW[i][j]) / s2 for j in range(n)]
                for i in range(n)
            ]
        for i in range(n):
            u = st.unif()
            if not latent:
                continue
            lo, hi = bounds_fn(i)
            if lo is None:
                continue
            cm = mu[i] - ssum(H[i][j] * (z[j] - mu[j]) for j in range(n) if j != i) / H[i][i]
            z[i] = _tnorm(cm, 1.0 / math.sqrt(H[i][i]), lo, hi, u)
        if cut_fn is not None:
            cut_fn(z, st)
        # collapsed order: rho | z (beta and, when free, sigma integrated), then sigma^2 | z, rho, then beta
        Wz = _mv(W, z)
        xz = [ssum(X[r][a] * z[r] for r in range(n)) for a in range(k)]
        xw = [ssum(X[r][a] * Wz[r] for r in range(n)) for a in range(k)]
        Mi = xtxi if exact else xtxi0
        pz, pw = _mv(Mi, xz), _mv(Mi, xw)
        c0 = ssum(v * v for v in z) - ssum(p * q for p, q in zip(xz, pz))
        c1 = ssum(p * q for p, q in zip(z, Wz)) - ssum(p * q for p, q in zip(xz, pw))
        c2 = ssum(v * v for v in Wz) - ssum(p * q for p, q in zip(xw, pw))
        rho = _draw_rho(grid, lndet, lnprior, c0, c1, c2, 0.0 if exact else (n - k) / 2.0, st.unif())
        Az = [a - rho * b for a, b in zip(z, Wz)]
        b0 = _mv(xtxi, [ssum(X[r][a] * Az[r] for r in range(n)) for a in range(k)])
        e = [Az[r] - ssum(X[r][a] * b0[a] for a in range(k)) for r in range(n)]
        if not sigma_fixed:
            # sigma^2 | rho, z with beta integrated out (flat priors): e'e / chi^2_{n-k}
            chi = ssum(st.norm() ** 2 for _ in range(n - k))
            s2 = ssum(v * v for v in e) / chi
        else:
            for _ in range(n - k):
                st.unif()
        zn = [st.norm() for _ in range(k)]
        beta = [b0[a] + math.sqrt(s2) * ssum(Lb[a][c] * zn[c] for c in range(a + 1)) for a in range(k)]
        if it >= burn_in:
            keep_b.append(beta)
            keep_r.append(rho)
            keep_s.append(s2)
            if cut_fn is not None:
                keep_c.append(list(cut_fn.cuts))
    return keep_b, keep_r, keep_s, keep_c


def _summ(keep_b, keep_r, keep_s=None):
    m = len(keep_r)
    k = len(keep_b[0])
    bm = [ssum(b[a] for b in keep_b) / m for a in range(k)]
    bs = [math.sqrt(ssum((b[a] - bm[a]) ** 2 for b in keep_b) / (m - 1)) for a in range(k)]
    rm = ssum(keep_r) / m
    out = {
        "beta": bm,
        "beta_sd": bs,
        "rho": rm,
        "rho_sd": math.sqrt(ssum((v - rm) ** 2 for v in keep_r) / (m - 1)),
        "rho_draws": keep_r,
        "beta_draws": keep_b,
    }
    if keep_s is not None:
        out["sigma2"] = ssum(keep_s) / m
    return out


def sar_probit_gibbs(y, X, W, ndraw=1000, burn_in=200, seed=0, a1=1.0, a2=1.0, prior_var=1e12, method="exact"):
    r"""Bayesian SAR probit ``z = rho W z + X beta + e``, ``y = 1(z > 0)`` (LeSage and Pace 2009, ch. 10).

    Gibbs sampler: each latent ``z_i`` from its univariate truncated normal
    full conditional under the precision ``H = (I - rho W)'(I - rho W)`` (one
    sweep per iteration, Geweke 1991); ``beta | z, rho ~ N(V X' (I - rho W)
    z, V)`` with ``V = (X'X + I / prior_var)^{-1}`` (prior ``beta ~ N(0,
    prior_var I)``); ``rho | z`` (``beta`` integrated out) by griddy Gibbs with
    density ``|I - rho W| exp(-q(rho) / 2)``, ``q(rho) = z'A'Az - (X'Az)' V
    (X'Az)``, times a Beta(a1, a2) prior on (-1, 1). ``method="lesage"`` uses
    instead the concentrated form ``|I - rho W| (e0 - rho ed)'(e0 - rho
    ed)^{-(n-k)/2}`` of LeSage and Pace (2009) and ``spatialprobit::sarprobit``,
    which treats the unit latent variance as unknown. ``X`` should contain the
    intercept column. Returns posterior means and standard deviations and
    the retained draws.

    References
    ----------
    LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, chapter 10.

    Wilhelm, S. and de Matos, M. G. (2013). Estimating spatial probit models in
    R. *The R Journal* 5, 130-143.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = sar_probit_gibbs([1, 1, 0, 0], [[1, 1.0], [1, 0.5], [1, -0.5], [1, -1.0]], W, ndraw=50, burn_in=10)
    >>> len(r.rho_draws), -1 < r.rho < 1
    (50, True)
    """
    yv = _vec(y)
    Xm, Wm = _mat(X), _mat(W)

    def bounds(i):
        return (0.0, math.inf) if yv[i] > 0 else (-math.inf, 0.0)

    if method not in ("exact", "lesage"):
        raise ValueError("method must be 'exact' or 'lesage'")
    kb, kr, _, _ = _latent_gibbs(
        yv, Xm, Wm, ndraw, burn_in, seed, a1, a2, bounds, True, prior_var=prior_var, exact=method == "exact"
    )
    return RichResult(payload=_summ(kb, kr))


def sar_tobit_gibbs(y, X, W, ndraw=1000, burn_in=200, seed=0, a1=1.0, a2=1.0):
    r"""Bayesian SAR Tobit: ``z = rho W z + X beta + e``, ``e ~ N(0, sigma^2 I)``, ``y = max(z, 0)``.

    Censored observations (``y <= 0``) are latent and drawn from their
    truncated normal full conditionals below 0 given all other ``z``; observed
    ones keep ``z = y``. ``beta`` and ``rho`` are drawn as in the SAR probit,
    and ``sigma^2 = e'e / chi^2_n`` (flat prior), following
    ``spatialprobit::sartobit`` (LeSage and Pace 2009, section 10.3).

    References
    ----------
    LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, section 10.3.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = sar_tobit_gibbs([1.2, 0.4, 0.0, 0.0], [[1, 1.0], [1, 0.5], [1, -0.5], [1, -1.0]], W, ndraw=40, burn_in=10)
    >>> r.sigma2 > 0
    True
    """
    yv = _vec(y)
    Xm, Wm = _mat(X), _mat(W)
    obs = [v if v > 0 else None for v in yv]

    def bounds(i):
        return (None, None) if yv[i] > 0 else (-math.inf, 0.0)

    kb, kr, ks, _ = _latent_gibbs(yv, Xm, Wm, ndraw, burn_in, seed, a1, a2, bounds, False, observed=obs)
    return RichResult(payload=_summ(kb, kr, ks))


def sar_ordered_probit_gibbs(y, X, W, ndraw=1000, burn_in=200, seed=0, a1=1.0, a2=1.0, prior_var=1e12, method="exact"):
    r"""Bayesian SAR ordered probit with categories ``1..J`` and cut points ``0 = phi_1 < ... < phi_{J-1}``.

    ``z = rho W z + X beta + e`` and ``y = j`` when ``phi_{j-1} < z <= phi_j``
    (``phi_0 = -inf``, ``phi_J = inf``). Latent ``z`` are drawn as in the SAR
    probit within their category interval, and each free cut point uniformly
    between the largest ``z`` of its lower category and the smallest ``z`` of
    its upper category (Albert and Chib 1993); ``beta`` and ``rho`` as in
    :func:`sar_probit_gibbs` (same ``prior_var`` and ``method``); ``X`` must
    include an intercept.

    References
    ----------
    Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
    polychotomous response data. *JASA* 88, 669-679.

    LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, section 10.2.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = sar_ordered_probit_gibbs([3, 2, 1, 1], [[1, 1.0], [1, 0.5], [1, -0.5], [1, -1.0]], W, ndraw=40, burn_in=10)
    >>> len(r.cutpoints)
    2
    """
    yv = [int(v) for v in _vec(y)]
    J = max(yv)
    Xm, Wm = _mat(X), _mat(W)

    class Cuts:
        def __init__(self):
            self.cuts = [0.0] + [float(j) for j in range(1, J - 1)]

        def __call__(self, z, st):
            for j in range(1, J - 1):
                lo = max([z[i] for i in range(len(z)) if yv[i] == j + 1] + [self.cuts[j - 1]])
                hi = min(
                    [z[i] for i in range(len(z)) if yv[i] == j + 2] + [self.cuts[j + 1] if j + 1 < J - 1 else math.inf]
                )
                u = st.unif()
                if hi > lo and hi < math.inf:
                    self.cuts[j] = lo + u * (hi - lo)

    cut = Cuts()

    def bounds(i):
        full = [-math.inf] + cut.cuts + [math.inf]
        return full[yv[i] - 1], full[yv[i]]

    if method not in ("exact", "lesage"):
        raise ValueError("method must be 'exact' or 'lesage'")
    kb, kr, _, kc = _latent_gibbs(
        yv, Xm, Wm, ndraw, burn_in, seed, a1, a2, bounds, True, cut_fn=cut, prior_var=prior_var, exact=method == "exact"
    )
    out = _summ(kb, kr)
    m = len(kc)
    out["cutpoints"] = [ssum(c[j] for c in kc) / m for j in range(J - 1)]
    return RichResult(payload=out)


def spatial_bayes_gibbs(y, X, W, model="lag", ndraw=1000, burn_in=200, seed=0, a1=1.0, a2=1.0):
    r"""Bayesian spatial lag, error and Durbin regressions by Gibbs sampling (flat priors).

    ``"lag"``: ``y = rho W y + X beta + e``; ``beta | rho, sigma^2`` normal,
    ``sigma^2 = e'e / chi^2_n`` and ``rho | y`` by griddy Gibbs with density
    ``|I - rho W| (e0 - rho ed)'(e0 - rho ed)^{-(n-k)/2}`` (``beta`` and
    ``sigma`` integrated out). ``"durbin"``: the lag model with ``W X`` (not
    the intercept) added. ``"error"``: ``y = X beta + u``, ``u = lambda W u + e``;
    ``beta`` by GLS, ``sigma^2`` as above and ``lambda | beta, sigma^2`` by
    griddy Gibbs with density ``|I - lambda W| exp(-(e - lambda W e)'(e - lambda W
    e) / (2 sigma^2))``. ``X`` must include the intercept column first.

    References
    ----------
    LeSage, J. P. and Pace, R. K. (2009). *Introduction to Spatial
    Econometrics*. CRC Press, chapter 5.

    Examples
    --------
    >>> W = [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]]
    >>> r = spatial_bayes_gibbs([2.0, 1.4, 0.3, -0.5], [[1, 1.0], [1, 0.5], [1, -0.5], [1, -1.0]], W, ndraw=50, burn_in=10)
    >>> len(r.beta)
    2
    """
    yv = _vec(y)
    Xm, Wm = _mat(X), _mat(W)
    n = len(yv)
    if model == "durbin":
        cols = [list(c) for c in zip(*Xm)][1:]
        wx = [_mv(Wm, c) for c in cols]
        Xm = [r + [c[i] for c in wx] for i, r in enumerate(Xm)]
    k = len(Xm[0])
    if model in ("lag", "durbin"):
        obs = [v for v in yv]

        def bounds(i):
            return (None, None)

        kb, kr, ks, _ = _latent_gibbs(yv, Xm, Wm, ndraw, burn_in, seed, a1, a2, bounds, False, observed=obs)
        return RichResult(payload=_summ(kb, kr, ks))
    if model != "error":
        raise ValueError("model must be 'lag', 'error' or 'durbin'")
    grid, lndet, lnprior = _setup(Wm, a1, a2)
    st = _Stream(seed, (ndraw + burn_in) * (n + k + 2))
    lam, s2 = 0.0, 1.0
    Wy = _mv(Wm, yv)
    WX = [list(c) for c in zip(*[_mv(Wm, list(c)) for c in zip(*Xm)])]
    kb, kr, ks = [], [], []
    for it in range(ndraw + burn_in):
        ys = [a - lam * b for a, b in zip(yv, Wy)]
        Xs = [[a - lam * b for a, b in zip(r, q)] for r, q in zip(Xm, WX)]
        xtxi = inverse([[ssum(r[a] * r[b] for r in Xs) for b in range(k)] for a in range(k)])
        b0 = _mv(xtxi, [ssum(Xs[r][a] * ys[r] for r in range(n)) for a in range(k)])
        L = _chol(xtxi)
        zn = [st.norm() for _ in range(k)]
        beta = [b0[a] + math.sqrt(s2) * ssum(L[a][c] * zn[c] for c in range(a + 1)) for a in range(k)]
        e = [ys[r] - ssum(Xs[r][a] * beta[a] for a in range(k)) for r in range(n)]
        s2 = ssum(v * v for v in e) / ssum(st.norm() ** 2 for _ in range(n))
        u = [yv[r] - ssum(Xm[r][a] * beta[a] for a in range(k)) for r in range(n)]
        Wu = _mv(Wm, u)
        c0, c1, c2 = ssum(v * v for v in u), ssum(p * q for p, q in zip(u, Wu)), ssum(v * v for v in Wu)
        den_lp = [lp - (c0 - 2 * r * c1 + r * r * c2) / (2 * s2) for r, lp in zip(grid, lnprior)]
        lam = _draw_rho(grid, lndet, den_lp, 1.0, 0.0, 0.0, 0.0, st.unif())
        if it >= burn_in:
            kb.append(beta)
            kr.append(lam)
            ks.append(s2)
    return RichResult(payload=_summ(kb, kr, ks))


def cheatsheet() -> str:
    return "sar_probit_gibbs / sar_tobit_gibbs / sar_ordered_probit_gibbs / spatial_bayes_gibbs -> Bayesian SAR models."
