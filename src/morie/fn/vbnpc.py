# morie.fn -- function file (rootcoder007/morie)
"""Variational Bayes for a Dirichlet process mixture (truncated stick-breaking).

SOURCE.  Blei, D.M. and Jordan, M.I. (2006), "Variational inference for
Dirichlet process mixtures", *Bayesian Analysis* 1(1):121-143,
doi:10.1214/06-BA104.

The DP mixture is written in Sethuraman's stick-breaking form,
V_t ~ Beta(1, alpha), pi_t(V) = V_t prod_{i<t} (1 - V_i), z_n ~ pi,
y_n | z_n ~ p(. | eta_{z_n}), and the variational family is the
TRUNCATED mean-field family of the paper's Section 4:

    q(V, eta, z) = prod_{t<K} q(V_t) prod_{t<=K} q(eta_t) prod_n q(z_n)

with V_K set to 1 so that pi_t = 0 for t > K.  The truncation is on the
variational distribution, not on the model -- the paper's point in
Section 4 -- so the fitted object is still a DP mixture.

Coordinate updates (paper's Eqs. 18-21):

    gamma_{t,1} = 1 + sum_n phi_{n,t}
    gamma_{t,2} = alpha + sum_n sum_{j>t} phi_{n,j}
    phi_{n,t} ∝ exp( E[log V_t] + sum_{i<t} E[log(1 - V_i)]
                     + E_q[ log p(y_n | eta_t) ] )

with E[log V_t] = psi(gamma_{t,1}) - psi(gamma_{t,1} + gamma_{t,2}) and
E[log(1 - V_t)] = psi(gamma_{t,2}) - psi(gamma_{t,1} + gamma_{t,2}).

COMPONENT MODEL.  Univariate Gaussian with KNOWN variance sigma2 and a
conjugate normal base measure G0 = N(m0, s0^2), so q(eta_t) = N(m_t,
s_t^2) is closed form:

    1/s_t^2 = 1/s0^2 + N_t/sigma2,   N_t = sum_n phi_{n,t}
    m_t     = ( m0/s0^2 + (sum_n phi_{n,t} y_n)/sigma2 ) * s_t^2
    E_q[log p(y|eta_t)] = -log(2 pi sigma2)/2
                          - ((y - m_t)^2 + s_t^2) / (2 sigma2).

An unknown-variance component would need a Normal-Gamma base measure;
that is not implemented and the omission is this implementation's scope
choice, stated rather than attributed.

INITIALISATION is deterministic -- the component means start at the K
type-7 quantiles of y at levels (t - 1/2)/K -- because a random start
would put the two language arms on different local optima and make a
1e-9 parity comparison meaningless.

The ELBO of the paper's Section 4 is evaluated every sweep and its
monotonicity is asserted (``elbo_monotone``); with K = 1 the fit
collapses to the exact conjugate normal posterior, which is the
closed-form anchor.
"""

from __future__ import annotations

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["vb_nonparametric"]


def _lbeta(a, b):
    return core.lgamma(a) + core.lgamma(b) - core.lgamma(a + b)


def vb_nonparametric(y, K_truncate=5, alpha=1.0, sigma2=1.0, m0=0.0,
                     s0=10.0, max_iter=100, tol=1e-10):
    """Truncated stick-breaking mean-field VB for a Gaussian DP mixture.

    Parameters
    ----------
    y : array-like
        Observations.
    K_truncate : int
        Truncation level K >= 1.
    alpha : float
        DP concentration, > 0.
    sigma2 : float
        Known component variance, > 0.
    m0, s0 : float
        Base measure N(m0, s0^2); ``s0`` > 0.
    max_iter : int
        Maximum coordinate sweeps.
    tol : float
        Stop when the ELBO moves by less than this.

    Returns
    -------
    RichResult
        ``phi`` (n-by-K responsibilities), ``m`` and ``s2`` (component
        posteriors), ``gamma1``, ``gamma2``, ``weights`` (E[pi_t]),
        ``elbo``, ``elbo_path``, ``elbo_monotone``, ``iterations``,
        ``converged``, ``n``, ``k``.

    Raises
    ------
    ValueError
        Empty ``y``, K < 1, or a non-positive alpha, sigma2, s0 or tol.

    References
    ----------
    Blei, D.M. and Jordan, M.I. (2006).  Bayesian Analysis
    1(1):121-143.  doi:10.1214/06-BA104.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("vb_nonparametric: y is empty")
    K = int(K_truncate)
    if K < 1:
        raise ValueError("vb_nonparametric: K_truncate must be at least 1")
    alpha = float(alpha)
    sigma2 = float(sigma2)
    s0 = float(s0)
    tol = float(tol)
    if alpha <= 0.0:
        raise ValueError("vb_nonparametric: alpha must be positive")
    if sigma2 <= 0.0:
        raise ValueError("vb_nonparametric: sigma2 must be positive")
    if s0 <= 0.0:
        raise ValueError("vb_nonparametric: s0 must be positive")
    if tol <= 0.0:
        raise ValueError("vb_nonparametric: tol must be positive")
    m0 = float(m0)
    p0 = 1.0 / (s0 * s0)
    # deterministic start: component means at the K type-7 quantiles
    m = [core.quantile7(yv, (t + 0.5) / K) for t in range(K)]
    s2 = [1.0 / p0] * K
    g1 = [1.0] * K
    g2 = [alpha] * K
    phi = [[1.0 / K] * K for _ in range(n)]
    path = []
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        # --- q(z): Eq. (21)
        elv = [0.0] * K
        el1v = [0.0] * K
        for t in range(K):
            dg = core.digamma(g1[t] + g2[t])
            elv[t] = core.digamma(g1[t]) - dg
            el1v[t] = core.digamma(g2[t]) - dg
        elv[K - 1] = 0.0
        el1v[K - 1] = 0.0
        cum = [0.0] * K
        acc = 0.0
        for t in range(K):
            cum[t] = acc
            acc += el1v[t]
        for i in range(n):
            sc = [0.0] * K
            best = None
            for t in range(K):
                lp = (-0.5 * math.log(2.0 * math.pi * sigma2)
                      - ((yv[i] - m[t]) ** 2 + s2[t]) / (2.0 * sigma2))
                sc[t] = elv[t] + cum[t] + lp
                if best is None or sc[t] > best:
                    best = sc[t]
            tot = 0.0
            for t in range(K):
                sc[t] = math.exp(sc[t] - best)
                tot += sc[t]
            for t in range(K):
                phi[i][t] = sc[t] / tot
        # --- q(V): Eqs. (18)-(19)
        Nt = [0.0] * K
        Sy = [0.0] * K
        for t in range(K):
            a = 0.0
            b = 0.0
            for i in range(n):
                a += phi[i][t]
                b += phi[i][t] * yv[i]
            Nt[t] = a
            Sy[t] = b
        tail = 0.0
        gt = [0.0] * K
        for t in range(K - 1, -1, -1):
            gt[t] = tail
            tail += Nt[t]
        for t in range(K):
            g1[t] = 1.0 + Nt[t]
            g2[t] = alpha + gt[t]
        # --- q(eta): conjugate normal
        for t in range(K):
            prec = p0 + Nt[t] / sigma2
            s2[t] = 1.0 / prec
            m[t] = (m0 * p0 + Sy[t] / sigma2) * s2[t]
        # --- ELBO
        elv = [0.0] * K
        el1v = [0.0] * K
        for t in range(K):
            dg = core.digamma(g1[t] + g2[t])
            elv[t] = core.digamma(g1[t]) - dg
            el1v[t] = core.digamma(g2[t]) - dg
        elv[K - 1] = 0.0
        el1v[K - 1] = 0.0
        cum = [0.0] * K
        acc = 0.0
        for t in range(K):
            cum[t] = acc
            acc += el1v[t]
        elbo = 0.0
        for t in range(K - 1):
            elbo += math.log(alpha) + (alpha - 1.0) * el1v[t]
            elbo -= (-_lbeta(g1[t], g2[t]) + (g1[t] - 1.0) * elv[t]
                     + (g2[t] - 1.0) * el1v[t])
        for t in range(K):
            elbo += (-0.5 * math.log(2.0 * math.pi * s0 * s0)
                     - ((m[t] - m0) ** 2 + s2[t]) / (2.0 * s0 * s0))
            elbo += 0.5 * (math.log(2.0 * math.pi * s2[t]) + 1.0)
        for i in range(n):
            for t in range(K):
                p = phi[i][t]
                if p > 0.0:
                    lp = (-0.5 * math.log(2.0 * math.pi * sigma2)
                          - ((yv[i] - m[t]) ** 2 + s2[t]) / (2.0 * sigma2))
                    elbo += p * (elv[t] + cum[t] + lp - math.log(p))
        path.append(elbo)
        if len(path) > 1 and abs(path[-1] - path[-2]) < tol:
            converged = True
            break
    ev = [g1[t] / (g1[t] + g2[t]) for t in range(K)]
    ev[K - 1] = 1.0
    w = [0.0] * K
    rem = 1.0
    for t in range(K):
        w[t] = ev[t] * rem
        rem *= (1.0 - ev[t])
    mono = True
    for i in range(1, len(path)):
        if path[i] < path[i - 1] - 1e-8:
            mono = False
    return RichResult(
        title="Variational Bayes for a Dirichlet process mixture",
        summary_lines=[("obs", n), ("truncation", K), ("ELBO", path[-1])],
        payload={
            "estimate": path[-1],
            "phi": phi,
            "m": m,
            "s2": s2,
            "gamma1": g1,
            "gamma2": g2,
            "weights": w,
            "elbo": path[-1],
            "elbo_path": path,
            "elbo_monotone": 1.0 if mono else 0.0,
            "iterations": it,
            "converged": 1.0 if converged else 0.0,
            "n": n,
            "k": K,
            "method": "Truncated stick-breaking mean-field VB for a DP mixture (Blei and Jordan 2006 Sec. 4)",
        },
    )


def cheatsheet():
    return "vbnpc: variational Bayes for a DP mixture, truncated stick-breaking (Blei & Jordan 2006)"
