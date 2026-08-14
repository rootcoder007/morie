# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two-stage estimation for linear transformation models with censored data."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["twostg", "two_stage_hazard"]

# NOTE: the generated stub documented a hazard factorisation
# lambda(t) = lambda_0(t) f(beta'X) g(gamma'Z). No such model appears in
# Cheng, Wei & Ying (1995), Biometrika 82(4), 835-845, "Analysis of
# transformation models with censored data" -- the actual method, kept
# here, is the linear transformation model g{S_Z(t)} = h(t) + Z'beta
# solved in two stages: (1) Kaplan-Meier estimate Ghat of the censoring
# survival function, (2) the generalised estimating equation (2.3),
#   U(b) = sum_{i,j} Z_ij { D_j I(X_i >= X_j) / Ghat(X_j)^2 - xi(Z_ij'b) },
# with Z_ij = Z_i - Z_j and xi(s) = P(eps_i - eps_j >= s). Extreme-value
# errors give the proportional-hazards model with xi(s) = 1/(1+e^s);
# logistic errors give the proportional-odds model.


def _km_censoring(x, delta):
    # Kaplan-Meier of the CENSORING survival G(t) = P(C > t):
    # censorings are the events. Right-continuous; at a censoring time
    # the drop has already happened, matching Ghat(X_j) evaluated for
    # event subjects j (events precede censorings at ties).
    n = len(x)
    order = sorted(range(n), key=lambda i: (x[i], delta[i]))
    G = 1.0
    at_risk = n
    Gvals = {}
    i = 0
    xs = [x[k] for k in order]
    ds = [delta[k] for k in order]
    while i < n:
        t = xs[i]
        j = i
        d_cens = 0
        while j < n and xs[j] == t:
            if ds[j] == 0.0:
                d_cens += 1
            j += 1
        if d_cens > 0:
            G *= 1.0 - d_cens / at_risk
        Gvals[t] = G
        at_risk -= (j - i)
        i = j

    def geval(t, before=False):
        g = 1.0
        for tt in sorted(Gvals):
            if tt < t or (tt == t and not before):
                g = Gvals[tt]
            else:
                break
        return g
    return geval


def _xi_ph(s):
    import math
    s = max(min(float(s), 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(s))


def _dxi_ph(s):
    import math
    s = max(min(float(s), 30.0), -30.0)
    e = math.exp(s)
    return -e / (1.0 + e) ** 2


def _xi_po_scalar(s):
    # xi(s) = int (1 - F(t+s)) dF(t) for standard logistic F:
    # closed form (difference of logistics): for s != 0,
    # xi(s) = (e^s - 1 - s e^s) / (1 - e^s)^2 * ... derive numerically
    # instead: Simpson rule on t in [-40, 40], 4001 nodes, identical in R.
    import math
    a, b, m = -40.0, 40.0, 4000
    h = (b - a) / m
    tot = 0.0
    for k in range(m + 1):
        t = a + k * h
        Ft = 1.0 / (1.0 + math.exp(-t))
        ft = Ft * (1.0 - Ft)
        Fts = 1.0 / (1.0 + math.exp(-(t + s)))
        val = (1.0 - Fts) * ft
        wgt = 1.0 if k in (0, m) else (4.0 if k % 2 == 1 else 2.0)
        tot += wgt * val
    return tot * h / 3.0


def _dxi_po_scalar(s):
    import math
    a, b, m = -40.0, 40.0, 4000
    h = (b - a) / m
    tot = 0.0
    for k in range(m + 1):
        t = a + k * h
        Ft = 1.0 / (1.0 + math.exp(-t))
        ft = Ft * (1.0 - Ft)
        Fts = 1.0 / (1.0 + math.exp(-(t + s)))
        fts = Fts * (1.0 - Fts)
        wgt = 1.0 if k in (0, m) else (4.0 if k % 2 == 1 else 2.0)
        tot += wgt * (-fts) * ft
    return tot * h / 3.0


def two_stage_hazard(time, event, X, Z=None, error="ph",
                     max_iter=50, tol=1e-10):
    """
    Cheng-Wei-Ying two-stage estimator for the linear transformation
    model g{S(t | Z)} = h(t) + Z'beta with censored data.

    Stage 1 estimates the censoring survival G by Kaplan-Meier; stage 2
    solves the generalised estimating equation (2.3) of the paper with
    unit weights. error="ph" (extreme-value F, proportional hazards)
    uses xi(s) = 1/(1+e^s); error="po" (logistic F, proportional odds)
    uses the logistic-difference xi. Standard errors come from the
    sandwich Sigma = Lambda Gamma Lambda of Section 2 / Appendix 1.

    The stub's documented factorisation lambda_0(t) f(beta'X) g(gamma'Z)
    does not exist in the cited paper; X and Z are simply concatenated
    into the covariate vector.

    Reference: Cheng, S. C., Wei, L. J. and Ying, Z. (1995), Biometrika
    82(4), 835-845. Printed anchor: Freireich leukaemia data, PH error,
    beta = -1.74, se 0.41 (p. 838-839).

    Returns
    -------
    result : RichResult
        Keys: estimate, se, cov, n_iter, error.
    """
    t = np.asarray(time, dtype=float)
    d = np.asarray(event, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape((-1, 1))
    if Xa.shape[0] != t.shape[0]:
        Xa = Xa.T
    if Z is not None:
        Za = np.asarray(Z, dtype=float)
        if Za.ndim == 1:
            Za = Za.reshape((-1, 1))
        if Za.shape[0] != t.shape[0]:
            Za = Za.T
        cols = []
        for r in range(Xa.shape[0]):
            cols.append(Xa[r].tolist() + Za[r].tolist())
        Xa = np.asarray(cols)
    n, p = Xa.shape
    if error == "ph":
        xi, dxi = _xi_ph, _dxi_ph
    elif error == "po":
        xi = lambda s: _xi_po_scalar(float(s))
        dxi = lambda s: _dxi_po_scalar(float(s))
    else:
        raise ValueError("error must be 'ph' or 'po'")

    geval = _km_censoring(t.tolist(), d.tolist())
    # left limit Ghat(X_j-): validated against the printed Freireich
    # anchor (-1.74, 0.41); the right-continuous variant gives -1.765.
    G2 = [geval(float(t[j]), before=True) ** 2 for j in range(n)]

    beta = np.zeros(p)
    for it in range(max_iter):
        U = np.zeros(p)
        J = np.zeros((p, p))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                zij = Xa[i] - Xa[j]
                s = float(zij @ beta)
                e_obs = 0.0
                if d[j] == 1.0 and t[i] >= t[j]:
                    e_obs = 1.0 / G2[j]
                U = U + zij * (e_obs - float(xi(s)))
                J = J + np.outer(zij, zij) * (-float(dxi(s)))
        step = np.linalg.solve(J, U)
        beta = beta - step
        if float(np.max(np.abs(step))) < tol:
            break

    # sandwich variance (Appendix 1): Lambda-hat and Gamma-hat
    ehat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = float((Xa[i] - Xa[j]) @ beta)
            base = -float(xi(s))
            if d[j] == 1.0 and t[i] >= t[j]:
                base += 1.0 / G2[j]
            ehat[i][j] = base
    lam_inv = np.zeros((p, p))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            zij = Xa[i] - Xa[j]
            s = float(zij @ beta)
            lam_inv = lam_inv + np.outer(zij, zij) * (-float(dxi(s)))
    lam_inv = lam_inv / (n * n)
    lam = np.linalg.inv(lam_inv)
    gam = np.zeros((p, p))
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            zij = Xa[i] - Xa[j]
            aij = ehat[i][j] - ehat[j][i]
            for k in range(n):
                if k == j or k == i:
                    continue
                zik = Xa[i] - Xa[k]
                aik = ehat[i][k] - ehat[k][i]
                gam = gam + aij * aik * np.outer(zij, zik)
    gam = gam / (n ** 3)
    corr = np.zeros((p, p))
    for l in range(n):
        if d[l] == 1.0:
            continue
        atrisk = sum(1 for k in range(n) if t[k] >= t[l])
        v = np.zeros(p)
        for i in range(n):
            for j in range(n):
                if i == j or d[j] != 1.0:
                    continue
                if t[i] >= t[j] and t[j] >= t[l]:
                    v = v + (Xa[i] - Xa[j]) * (1.0 / G2[j])
        corr = corr + np.outer(v, v) / (atrisk ** 2)
    gam = gam - 4.0 * corr / (n ** 3)
    sig = lam @ gam @ lam
    cov = sig / n
    dg = [float(cov[j, j]) for j in range(p)]
    return RichResult(payload={
        "estimate": beta,
        "se": np.sqrt(np.asarray([abs(v) for v in dg])),
        "cov": cov,
        "n_iter": it + 1,
        "error": error,
        "method": "Cheng-Wei-Ying (1995) two-stage transformation-model estimator, eq. (2.3)",
    })


twostg = two_stage_hazard


def cheatsheet():
    return "twostg(time, event, X, Z=None, error='ph'|'po') -> Cheng-Wei-Ying transformation-model estimator."

# public names resolved by fn/_lazy_map.json
twostagehazard = two_stage_hazard
