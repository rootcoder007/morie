"""HLM random-effects covariance (T matrix) estimation (Raudenbush & Bryk 2002)."""

import math

from ._richresult import RichResult

__all__ = ["hlmgr", "hlm_tau_matrix"]


def _inv(a):
    k = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(k)]
         for i, row in enumerate(a)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            raise ValueError("singular matrix")
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        for j in range(2 * k):
            m[c][j] /= d
        for r in range(k):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                for j in range(2 * k):
                    m[r][j] -= f * m[c][j]
    return [row[k:] for row in m]


def _eig_clip_psd(a):
    # symmetric 2x2/general: clip negative eigenvalues to 0 via Jacobi
    k = len(a)
    v = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    m = [row[:] for row in a]
    for _sweep in range(100):
        off = max((abs(m[p][q]), p, q) for p in range(k)
                  for q in range(p + 1, k)) if k > 1 else (0.0, 0, 0)
        if off[0] < 1e-14:
            break
        p, q = off[1], off[2]
        theta = 0.5 * math.atan2(2.0 * m[p][q], m[q][q] - m[p][p])
        c, s = math.cos(theta), math.sin(theta)
        for i in range(k):
            mp, mq = m[i][p], m[i][q]
            m[i][p] = c * mp - s * mq
            m[i][q] = s * mp + c * mq
        for j in range(k):
            mp, mq = m[p][j], m[q][j]
            m[p][j] = c * mp - s * mq
            m[q][j] = s * mp + c * mq
        for i in range(k):
            vp, vq = v[i][p], v[i][q]
            v[i][p] = c * vp - s * vq
            v[i][q] = s * vp + c * vq
    lam = [max(m[i][i], 0.0) for i in range(k)]
    return [[sum(v[i][t] * lam[t] * v[j][t] for t in range(k))
             for j in range(k)] for i in range(k)]


def hlmgr(betas, V=None):
    """
    Method-of-moments T (tau) matrix for HLM random coefficients.

    Raudenbush & Bryk (2002), Ch. 3: the OLS level-1 coefficient
    estimates beta_hat_j disperse as

        Var(beta_hat_j) = T + V_j
        ("parameter dispersion + error dispersion", their Eq. 3.28),

    so the level-2 variance-covariance matrix of the random effects
    is estimated by T_hat = S - V_bar, the sample covariance of the
    beta_hat_j minus the average sampling covariance, projected to
    the nearest positive semidefinite matrix (negative eigenvalues
    clipped).  Also returned per group: the multivariate reliability
    Lambda_j = T (T + V_j)^{-1} (their Eq. 3.57) and the empirical
    Bayes shrinkage estimates
    beta*_j = Lambda_j beta_hat_j + (I - Lambda_j) gamma_hat with
    gamma_hat the grand mean (their composite estimator discussion:
    unreliable beta_hat_j are pulled toward the level-2 prediction).

    Sources
    -------
    Raudenbush, S. W. & Bryk, A. S. (2002). *Hierarchical Linear
    Models: Applications and Data Analysis Methods*, 2nd ed., Sage,
    Ch. 3, Eqs. 3.28 and 3.57 and the shrinkage discussion (local
    copy fetched-wave3/
    Hierarchical_Linear_Models_Applications_and_Data_Analysis_Methods.pdf).

    Parameters
    ----------
    betas : matrix (J groups x q coefficients)
        Per-group OLS coefficient estimates.
    V : sequence of q x q matrices, optional
        Per-group sampling covariance of beta_hat_j (default all
        zero, in which case T_hat is the plain sample covariance).

    Returns
    -------
    RichResult
        Keys: tau (T_hat), gamma (grand means), reliabilities
        (Lambda_j), shrunken (beta*_j), s_total (sample covariance).
    """
    B = [[float(v) for v in row] for row in betas]
    J = len(B)
    if J < 3:
        raise ValueError("need at least three groups")
    q = len(B[0])
    if any(len(r) != q for r in B):
        raise ValueError("beta rows must have equal length")
    if V is None:
        Vs = [[[0.0] * q for _ in range(q)] for _ in range(J)]
    else:
        Vs = [[[float(x) for x in row] for row in vj] for vj in V]
        if len(Vs) != J:
            raise ValueError("need one V_j per group")
    gamma = [sum(B[j][a] for j in range(J)) / J for a in range(q)]
    S = [[sum((B[j][a] - gamma[a]) * (B[j][b] - gamma[b])
              for j in range(J)) / (J - 1) for b in range(q)]
         for a in range(q)]
    vbar = [[sum(Vs[j][a][b] for j in range(J)) / J for b in range(q)]
            for a in range(q)]
    raw = [[S[a][b] - vbar[a][b] for b in range(q)] for a in range(q)]
    tau = _eig_clip_psd(raw)
    lams = []
    shrunk = []
    for j in range(J):
        tv = [[tau[a][b] + Vs[j][a][b] for b in range(q)]
              for a in range(q)]
        tvi = _inv(tv)
        lam = [[sum(tau[a][t] * tvi[t][b] for t in range(q))
                for b in range(q)] for a in range(q)]
        lams.append(lam)
        bs = [sum(lam[a][t] * B[j][t] for t in range(q))
              + gamma[a] - sum(lam[a][t] * gamma[t] for t in range(q))
              for a in range(q)]
        shrunk.append(bs)
    return RichResult(payload={
        "tau": tau,
        "gamma": gamma,
        "reliabilities": lams,
        "shrunken": shrunk,
        "s_total": S,
        "J": J,
        "method": "HLM T matrix, MoM (R&B 2002 Eqs. 3.28, 3.57)",
    })


# long descriptive alias (stub-era name)
hlm_tau_matrix = hlmgr


def cheatsheet():
    return "hlmgr: T = S - Vbar (PSD-clipped); Lambda = T(T+V)^-1; EB shrinkage"
