"""Network psychometrics via the graphical lasso (Epskamp 2018; ESL Alg. 17.2)."""

import math

from ._richresult import RichResult

__all__ = ["netcms", "network_psychometrics"]


def _soft(x, t):
    # ESL Eq. 17.27
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


def _solve_lasso(v, s12, lam, beta, tol, maxit):
    # cyclical coordinate descent, ESL Eq. 17.26
    p1 = len(s12)
    for _ in range(maxit):
        delta = 0.0
        for j in range(p1):
            r = s12[j] - sum(v[k][j] * beta[k] for k in range(p1)
                             if k != j)
            new = _soft(r, lam) / v[j][j]
            delta = max(delta, abs(new - beta[j]))
            beta[j] = new
        if delta < tol:
            break
    return beta


def netcms(data=None, S=None, lam=0.1, tol=1e-8, maxit=500):
    """
    Gaussian graphical model by the graphical lasso.

    Estimates a sparse inverse covariance (precision) matrix Theta by
    the graphical lasso of Friedman, Hastie & Tibshirani (2008), as
    given in ESL Algorithm 17.2: initialize W = S + lambda I (the
    diagonal of W stays fixed); cycle over variables j, solving the
    modified lasso system W11 beta - s12 + lambda Sign(beta) = 0 by
    cyclical coordinate descent (Eq. 17.26 with soft-threshold
    Eq. 17.27) and updating w12 = W11 beta_hat; finally recover
    theta_22 = 1/(w22 - w12' beta_hat) and theta_12 = -beta_hat
    theta_22.  The nonzero pattern of Theta is the estimated
    conditional-independence graph (ESL Eq. 17.1) -- in network
    psychometrics the partial-correlation network of Epskamp et al.
    At the optimum the KKT subgradient condition
    |[Theta^{-1} - S]_jk| <= lambda holds off-diagonally (Eq. 17.22).

    Sources
    -------
    Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements
    of Statistical Learning*, 2nd ed., Springer, Sec. 17.3.2,
    Algorithm 17.2, Eqs. 17.22-17.27 (local PDF:
    WD_BLACK/library/pdf/BookAdvanced_elementsofstatisticallearning.pdf).
    Friedman, J., Hastie, T. & Tibshirani, R. (2008). Sparse inverse
    covariance estimation with the graphical lasso. *Biostatistics*,
    9, 432-441.
    Epskamp, S., Borsboom, D. & Fried, E. I. (2018). Estimating
    psychological networks and their accuracy. *Behavior Research
    Methods*, 50, 195-212 (the network-psychometrics use, as cited
    by the stub).

    Parameters
    ----------
    data : sequence of rows, optional
        Observations; the sample covariance is used as S.
    S : sequence of rows, optional
        Covariance matrix directly (overrides data).
    lam : float
        L1 penalty lambda >= 0.
    tol, maxit : float, int
        Coordinate-descent tolerances.

    Returns
    -------
    RichResult
        Keys: precision (Theta), covariance_fit (W), adjacency,
        partial_correlations, n_edges, lam.
    """
    if S is None:
        if data is None:
            raise ValueError("provide data or S")
        rows = [[float(v) for v in r] for r in data]
        n = len(rows)
        p = len(rows[0])
        mu = [sum(r[j] for r in rows) / n for j in range(p)]
        S = [[sum((r[a] - mu[a]) * (r[b] - mu[b]) for r in rows) / n
              for b in range(p)] for a in range(p)]
    else:
        S = [[float(v) for v in r] for r in S]
        p = len(S)
    lam = float(lam)
    if lam < 0:
        raise ValueError("lam must be non-negative")
    W = [[S[a][b] + (lam if a == b else 0.0) for b in range(p)]
         for a in range(p)]
    betas = [[0.0] * (p - 1) for _ in range(p)]
    for _cycle in range(maxit):
        w_old = [row[:] for row in W]
        for j in range(p):
            idx = [k for k in range(p) if k != j]
            v = [[W[a][b] for b in idx] for a in idx]
            s12 = [S[a][j] for a in idx]
            beta = _solve_lasso(v, s12, lam, betas[j], tol, maxit)
            w12 = [sum(v[r][c] * beta[c] for c in range(p - 1))
                   for r in range(p - 1)]
            for t, a in enumerate(idx):
                W[a][j] = w12[t]
                W[j][a] = w12[t]
        diff = max(abs(W[a][b] - w_old[a][b])
                   for a in range(p) for b in range(p))
        if diff < tol:
            break
    theta = [[0.0] * p for _ in range(p)]
    for j in range(p):
        idx = [k for k in range(p) if k != j]
        beta = betas[j]
        w12b = sum(W[a][j] * beta[t] for t, a in enumerate(idx))
        t22 = 1.0 / (W[j][j] - w12b)
        theta[j][j] = t22
        for t, a in enumerate(idx):
            theta[a][j] = -beta[t] * t22
    # symmetrize (numerically) and derive network quantities
    for a in range(p):
        for b in range(a + 1, p):
            v = 0.5 * (theta[a][b] + theta[b][a])
            theta[a][b] = theta[b][a] = v
    adj = [[1 if a != b and abs(theta[a][b]) > 1e-10 else 0
            for b in range(p)] for a in range(p)]
    pcor = [[(-theta[a][b] / math.sqrt(theta[a][a] * theta[b][b])
              if a != b else 1.0) for b in range(p)] for a in range(p)]
    n_edges = sum(adj[a][b] for a in range(p)
                  for b in range(a + 1, p))
    return RichResult(payload={
        "precision": theta,
        "covariance_fit": W,
        "adjacency": adj,
        "partial_correlations": pcor,
        "n_edges": n_edges,
        "lam": lam,
        "method": "graphical lasso (ESL Alg. 17.2; Friedman 2008)",
    })


# long descriptive alias (stub-era name)
network_psychometrics = netcms


def cheatsheet():
    return "netcms: glasso W=S+lam I; lasso on W11,s12; theta from beta"
