# morie.fn -- function file (rootcoder007/morie)
"""Nyblom-Hansen joint parameter stability test."""

from __future__ import annotations

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["nyblom_hansen_stability"]

# Hansen (1992), Table 1: asymptotic critical values for L_c, indexed by
# degrees of freedom m+1 = 1..20, at 1%, 2.5%, 5%, 7.5%, 10%, 20%.
_LC_LEVELS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.20)
_LC_TABLE = (
    (0.748, 0.593, 0.470, 0.398, 0.353, 0.243),
    (1.07, 0.898, 0.749, 0.670, 0.610, 0.469),
    (1.35, 1.16, 1.01, 0.913, 0.846, 0.679),
    (1.60, 1.39, 1.24, 1.14, 1.07, 0.883),
    (1.88, 1.63, 1.47, 1.36, 1.28, 1.08),
    (2.12, 1.89, 1.68, 1.58, 1.49, 1.28),
    (2.35, 2.10, 1.90, 1.78, 1.69, 1.46),
    (2.59, 2.33, 2.11, 1.99, 1.89, 1.66),
    (2.82, 2.55, 2.32, 2.19, 2.10, 1.85),
    (3.05, 2.76, 2.54, 2.40, 2.29, 2.03),
    (3.27, 2.99, 2.75, 2.60, 2.49, 2.22),
    (3.51, 3.18, 2.96, 2.81, 2.69, 2.41),
    (3.69, 3.39, 3.15, 3.00, 2.89, 2.59),
    (3.90, 3.60, 3.34, 3.19, 3.08, 2.77),
    (4.07, 3.81, 3.54, 3.38, 3.26, 2.95),
    (4.30, 4.01, 3.75, 3.58, 3.46, 3.14),
    (4.51, 4.21, 3.95, 3.77, 3.64, 3.32),
    (4.73, 4.40, 4.14, 3.96, 3.83, 3.50),
    (4.92, 4.60, 4.33, 4.16, 4.03, 3.69),
    (5.13, 4.79, 4.52, 4.36, 4.22, 3.86),
)


def lccritical(df):
    """Hansen (1992) Table 1 critical values for ``df`` = m+1 parameters."""
    df = int(df)
    if df < 1 or df > len(_LC_TABLE):
        raise ValueError("Hansen's Table 1 covers 1 to 20 degrees of freedom")
    return {lvl: _LC_TABLE[df - 1][i] for i, lvl in enumerate(_LC_LEVELS)}


def nyblom_hansen_stability(y, X, add_intercept=True, variance=True):
    """Nyblom-Hansen test that every OLS parameter is stable.

    Fit ``y = X b + e`` by least squares and form the first-order
    conditions

        ``f_it = x_it e_t``   for the m regression parameters,
        ``f_{m+1,t} = e_t^2 - sigma^2``   for the error variance,

    whose cumulative sums ``S_t = sum_{s<=t} f_s`` are zero at ``t = n``
    by construction.  With ``V = sum_t f_t f_t'`` the joint statistic is

        ``L_c = (1/n) sum_{t=1}^{n} S_t' V^{-1} S_t``

    and the individual statistics are ``L_i = (1/n) sum_t S_it^2 /
    V_ii``.  Large values reject stability.

    Because ``S_n = 0``, the cumulative sums behave like a tied-down
    random walk under the null, which is why the limit is a Brownian
    bridge and the critical values depend only on the number of
    parameters tested -- not on ``X``, and not on any nuisance
    parameter.  Including the variance score is what gives the test
    power against a shift in ``sigma^2``, which the coefficient-only
    version is blind to.

    Parameters
    ----------
    y : array-like
        Response, in the order stability is tested against.
    X : array-like
        ``n x p`` regressors, ordered the same way.
    add_intercept : bool
        Prepend a column of ones.
    variance : bool
        Include the error-variance score, giving ``df = m + 1``.

    Returns
    -------
    RichResult
        ``statistic`` (L_c), ``df``, ``individual`` (the L_i),
        ``critical`` (Hansen's Table 1 row), ``n``, ``method``.

    References
    ----------
    Nyblom (1989), Testing for the constancy of parameters over time,
    JASA 84:223-230; Hansen (1992), Testing for parameter instability in
    linear models, Journal of Policy Modeling 14:517-533.  Hansen's
    paper was fetched in full from his own page
    (users.ssc.wisc.edu/~bhansen/papers/jpm_92.pdf): eqs (3)-(4) give
    the scores, (9)-(10) give ``L_c = (1/n) sum S_t' V^{-1} S_t`` with
    ``V = sum f_t f_t'``, and Table 1 -- reproduced verbatim above -- is
    the asymptotic critical-value table, sourced there from Hansen
    (1990).  No p-value is returned: Table 1 gives six points and
    interpolating a p-value between them would invent precision the
    published table does not carry.  The equivalent in Zeileis's
    ``strucchange`` is ``sctest(type = "Nyblom-Hansen")``, i.e. the
    Score-CUSUM process under the ``meanL2`` functional; note that
    strucchange averages over the ``n+1`` points of the empirical
    process including the zero at ``t = 0``, so its value is ``n/(n+1)``
    times the ``L_c`` of Hansen's eq (9) returned here.
    """
    y = T.vec(y)
    Xm = T.mat(X)
    n = len(y)
    if len(Xm) != n:
        raise ValueError("X must have one row per element of y")
    if add_intercept:
        Xm = [[1.0] + row for row in Xm]
    p = len(Xm[0])
    if n <= p + 1:
        raise ValueError("need more observations than parameters")
    beta, fitted, e, _ = T.olsfit(Xm, y)
    sigma2 = sum(ei * ei for ei in e) / n
    f = []
    for t in range(n):
        row = [Xm[t][j] * e[t] for j in range(p)]
        if variance:
            row.append(e[t] * e[t] - sigma2)
        f.append(row)
    k = len(f[0])
    V = [[sum(f[t][a] * f[t][b] for t in range(n)) for b in range(k)] for a in range(k)]
    S = []
    acc = [0.0] * k
    for t in range(n):
        acc = [acc[j] + f[t][j] for j in range(k)]
        S.append(acc[:])
    ident = [[1.0 if a == b else 0.0 for b in range(k)] for a in range(k)]
    Vinv = _solve(V, ident)
    lc = 0.0
    for t in range(n):
        st = S[t]
        for a in range(k):
            va = sum(Vinv[a][b] * st[b] for b in range(k))
            lc += st[a] * va
    lc /= n
    indiv = []
    for a in range(k):
        vaa = V[a][a]
        indiv.append(sum(S[t][a] ** 2 for t in range(n)) / (n * vaa) if vaa > 0 else float("nan"))
    return RichResult(
        payload={
            "statistic": float(lc),
            "df": int(k),
            "individual": indiv,
            "critical": lccritical(k),
            "n": int(n),
            "method": "Nyblom-Hansen joint parameter stability test",
        }
    )


def _solve(A, B):
    """Solve ``A Z = B`` by Gauss-Jordan with partial pivoting."""
    k = len(A)
    aug = [A[i][:] + B[i][:] for i in range(k)]
    m = len(aug[0])
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(aug[r][c]))
        if abs(aug[piv][c]) < 1e-300:
            raise ValueError("singular score covariance matrix")
        aug[c], aug[piv] = aug[piv], aug[c]
        d = aug[c][c]
        aug[c] = [v / d for v in aug[c]]
        for r in range(k):
            if r == c:
                continue
            fct = aug[r][c]
            if fct != 0.0:
                aug[r] = [aug[r][j] - fct * aug[c][j] for j in range(m)]
    return [row[k:] for row in aug]


def cheatsheet():
    return "nyblom_hansen_stability(y, X): L_c = (1/n) sum S_t' V^-1 S_t."


# compact alias per ledger/NAMING.md
nyblomhansen = nyblom_hansen_stability
