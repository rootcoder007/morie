# morie.fn -- function file (rootcoder007/morie)
r"""Multiple imputation by chained equations, then a survival model.

Dropping the incomplete rows is not a neutral act. Complete-case
analysis is unbiased only when the missingness does not depend on
anything -- and covariates in a cohort are almost never missing that
way. Multiple imputation by chained equations replaces each missing
entry, one variable at a time, from a regression on all the others,
cycling until the fills stop moving, and does the whole thing ``m``
times so that the *uncertainty about the fills* survives into the
standard error instead of being quietly discarded.

Each completed dataset is then fitted with a Cox proportional-hazards
model (Breslow's handling of ties) and the ``m`` fits are combined by
Rubin's rules:

.. math:: \bar Q = \tfrac1m\sum Q_\ell, \quad
          T = \bar U + \Bigl(1+\tfrac1m\Bigr)B, \quad
          B = \tfrac{1}{m-1}\sum (Q_\ell-\bar Q)^2 .

The between-imputation term :math:`B` is the whole point: it is zero
when nothing was missing, and the reported ``fraction_missing_info``
says how much of the final variance came from not knowing the fills.
The degrees of freedom use the Barnard-Rubin small-sample correction,
which matters exactly when the cohort is small enough for the
imputation to be doing real work.

The imputations are drawn from a deterministic low-discrepancy normal
sequence rather than a random number generator. The draws are genuine
-- each imputation differs, and ``B`` is positive whenever data are
missing -- but the result is byte-reproducible across languages and
across runs, which is what lets a published analysis be re-run.

A complete-case fit is computed alongside and returned, because the
comparison between the two is the evidence that the imputation changed
anything.

References
----------
van Buuren, S. (2018) *Flexible Imputation of Missing Data*, 2nd ed.,
CRC Press, Ch. 3 (the ``norm.nob`` univariate imputation used here) and
Ch. 4 (the chained-equations algorithm and its convergence),
doi:10.1201/9780429492259.

van Buuren, S. and Groothuis-Oudshoorn, K. (2011) "mice: Multivariate
imputation by chained equations in R", *Journal of Statistical Software*
**45**(3), 1-67, doi:10.18637/jss.v045.i03.

Rubin, D. B. (1987) *Multiple Imputation for Nonresponse in Surveys*,
Wiley, Ch. 3, doi:10.1002/9780470316696.

Barnard, J. and Rubin, D. B. (1999) "Small-sample degrees of freedom
with multiple imputation", *Biometrika* **86**(4), 948-955,
doi:10.1093/biomet/86.4.948.

Cox, D. R. (1972) "Regression models and life-tables", *Journal of the
Royal Statistical Society B* **34**(2), 187-220.

Breslow, N. E. (1974) "Covariance analysis of censored survival data",
*Biometrics* **30**(1), 89-99, doi:10.2307/2529620.

White, I. R. and Royston, P. (2009) "Imputing missing covariate values
for the Cox model", *Statistics in Medicine* **28**(15), 1982-1998,
doi:10.1002/sim.3618.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["chained_imputation"]

_EPS = 1e-12
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
           59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]


def _is_missing(v):
    if v is None:
        return True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True
    return f != f                       # NaN is the only value unequal to itself


def _ols(X, y, ridge_rel=1e-8):
    """Least squares with a ridge scaled to the design, plus sigma."""
    n = len(y)
    p = len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)]
         for a in range(p)]
    scale = sum(A[a][a] for a in range(p)) / p
    for a in range(p):
        A[a][a] += ridge_rel * max(scale, _EPS)
    rhs = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    beta = k.cholsolve(A, rhs)
    fit = [sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    dof = max(n - p, 1)
    sig2 = sum((y[i] - fit[i]) ** 2 for i in range(n)) / dof
    return beta, math.sqrt(max(sig2, 0.0))


def _cox_breslow(t, e, X, max_iter=100, tol=1e-10):
    """Cox partial likelihood, Breslow ties, Newton-Raphson."""
    n = len(t)
    p = len(X[0]) if n else 0
    if p == 0:
        raise ValueError("sschin: the Cox model needs at least one covariate")
    order = sorted(range(n), key=lambda i: (-t[i], i))
    beta = [0.0] * p
    it = 0
    converged = False
    info = [[0.0] * p for _ in range(p)]
    ll = 0.0
    for it in range(1, max_iter + 1):
        ll = 0.0
        grad = [0.0] * p
        info = [[0.0] * p for _ in range(p)]
        s0 = 0.0
        s1 = [0.0] * p
        s2 = [[0.0] * p for _ in range(p)]
        pos = 0
        while pos < n:
            # every subject with this time enters the risk set together
            tt = t[order[pos]]
            grp = []
            while pos < n and t[order[pos]] == tt:
                grp.append(order[pos])
                pos += 1
            for i in grp:
                z = sum(X[i][a] * beta[a] for a in range(p))
                w = math.exp(max(-500.0, min(500.0, z)))
                s0 += w
                for a in range(p):
                    s1[a] += w * X[i][a]
                    for b in range(p):
                        s2[a][b] += w * X[i][a] * X[i][b]
            d = [i for i in grp if e[i] > 0.5]
            if not d:
                continue
            dk = float(len(d))
            for i in d:
                ll += sum(X[i][a] * beta[a] for a in range(p))
                for a in range(p):
                    grad[a] += X[i][a]
            ll -= dk * math.log(max(s0, 1e-300))
            for a in range(p):
                grad[a] -= dk * s1[a] / max(s0, 1e-300)
                for b in range(p):
                    info[a][b] += dk * (s2[a][b] / max(s0, 1e-300)
                                        - s1[a] * s1[b]
                                        / max(s0 * s0, 1e-300))
        step = k.cholsolve([[info[a][b] + (1e-10 if a == b else 0.0)
                             for b in range(p)] for a in range(p)], grad)
        beta = [beta[a] + step[a] for a in range(p)]
        if max(abs(v) for v in step) < tol:
            converged = True
            break
    cov = [[0.0] * p for _ in range(p)]
    Ir = [[info[a][b] + (1e-10 if a == b else 0.0) for b in range(p)]
          for a in range(p)]
    for a in range(p):
        e_a = [0.0] * p
        e_a[a] = 1.0
        col = k.cholsolve(Ir, e_a)
        for b in range(p):
            cov[b][a] = col[b]
    return beta, [max(cov[a][a], 0.0) for a in range(p)], ll, it, converged


def _t_quantile(pq, df):
    """Two-sided Student-t quantile by Cornish-Fisher on the normal."""
    z = k.qnorm(pq)
    if df > 1e8:
        return z
    g1 = (z ** 3 + z) / 4.0
    g2 = (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / 96.0
    g3 = (3.0 * z ** 7 + 19.0 * z ** 5 + 17.0 * z ** 3 - 15.0 * z) / 384.0
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def chained_imputation(time, event, X, mi_iter=5, cycles=10, ties="breslow"):
    r"""MICE on the covariates, a Cox fit per imputation, Rubin pooling.

    Parameters
    ----------
    time, event : array-like
        Follow-up time and the event indicator (1 = event, 0 = censored).
    X : array-like, shape ``(n, p)``
        Covariates. Missing entries are ``None`` or NaN.
    mi_iter : int
        Number of imputations ``m``. Rubin's between-imputation variance
        needs at least two.
    cycles : int
        Chained-equation cycles per imputation.

    Returns
    -------
    RichResult
        Pooled ``coefficients``, ``std_error``, ``df``, confidence
        interval, ``fraction_missing_info``, the per-imputation fits, and
        the complete-case fit for comparison.
    """
    tv = [float(v) for v in k.vec(time)]
    ev = [float(v) for v in k.vec(event)]
    # NOT k.mat: it coerces to float, and a missing entry is exactly what
    # cannot be coerced -- the rows are kept raw and tested for missingness
    Xr = [list(row) if isinstance(row, (list, tuple)) else [row]
          for row in X]
    n = len(tv)
    if n == 0:
        raise ValueError("sschin: no observations")
    if len(ev) != n or len(Xr) != n:
        raise ValueError("sschin: time, event and X must agree in length "
                         "(%d, %d, %d)" % (n, len(ev), len(Xr)))
    p = len(Xr[0])
    if any(len(r) != p for r in Xr):
        raise ValueError("sschin: every row of X must have %d columns" % p)
    m = int(mi_iter)
    if m < 2:
        raise ValueError("sschin: multiple imputation needs at least two "
                         "imputations -- the between-imputation variance is "
                         "undefined for m = 1")
    if m > len(_PRIMES):
        raise ValueError("sschin: at most %d imputations" % len(_PRIMES))
    if ties != "breslow":
        raise ValueError("sschin: only the Breslow handling of ties is "
                         "implemented, got %r" % (ties,))
    if not any(v > 0.5 for v in ev):
        raise ValueError("sschin: no events -- the partial likelihood is flat")

    miss = [[_is_missing(Xr[i][a]) for a in range(p)] for i in range(n)]
    obs = [[0.0 if miss[i][a] else float(Xr[i][a]) for a in range(p)]
           for i in range(n)]
    n_missing = sum(1 for i in range(n) for a in range(p) if miss[i][a])
    cols_missing = [a for a in range(p)
                    if any(miss[i][a] for i in range(n))]
    for a in cols_missing:
        if all(miss[i][a] for i in range(n)):
            raise ValueError("sschin: column %d is missing for every "
                             "observation and cannot be imputed" % a)

    colmean = []
    for a in range(p):
        ok = [obs[i][a] for i in range(n) if not miss[i][a]]
        colmean.append(sum(ok) / len(ok) if ok else 0.0)

    ests, vars_, per = [], [], []
    for ell in range(m):
        # one deterministic normal stream per imputation; different base
        # means genuinely different draws, so B is not degenerate
        draws = k.normdraws(max(n_missing * cycles, 1), _PRIMES[ell])
        pos = 0
        F = [[colmean[a] if miss[i][a] else obs[i][a] for a in range(p)]
             for i in range(n)]
        for _c in range(int(cycles)):
            for a in cols_missing:
                rows_obs = [i for i in range(n) if not miss[i][a]]
                rows_mis = [i for i in range(n) if miss[i][a]]
                others = [b for b in range(p) if b != a]
                # the outcome belongs in the imputation model: imputing a
                # covariate without it biases the fitted hazard ratio
                # towards the null (White & Royston 2009)
                def row(i):
                    return ([1.0] + [F[i][b] for b in others]
                            + [ev[i], math.log(max(tv[i], 1e-12))])
                Xo = [row(i) for i in rows_obs]
                yo = [F[i][a] for i in rows_obs]
                if len(rows_obs) <= len(Xo[0]):
                    beta = [sum(yo) / len(yo)] + [0.0] * (len(Xo[0]) - 1)
                    sig = 0.0
                else:
                    beta, sig = _ols(Xo, yo)
                for i in rows_mis:
                    rr = row(i)
                    mu = sum(rr[u] * beta[u] for u in range(len(rr)))
                    F[i][a] = mu + sig * draws[pos % len(draws)]
                    pos += 1
        b, v, ll, it, cv = _cox_breslow(tv, ev, F)
        ests.append(b)
        vars_.append(v)
        per.append({"coefficients": b, "variance": v, "loglik": ll,
                    "iterations": it, "converged": cv})

    qbar = [sum(ests[ell][a] for ell in range(m)) / m for a in range(p)]
    ubar = [sum(vars_[ell][a] for ell in range(m)) / m for a in range(p)]
    B = [sum((ests[ell][a] - qbar[a]) ** 2 for ell in range(m)) / (m - 1.0)
         for a in range(p)]
    T = [ubar[a] + (1.0 + 1.0 / m) * B[a] for a in range(p)]
    se = [math.sqrt(max(v, 0.0)) for v in T]
    riv, fmi, df = [], [], []
    n_events = sum(1 for v in ev if v > 0.5)
    dfcom = max(n_events - p, 1)
    for a in range(p):
        r = (1.0 + 1.0 / m) * B[a] / max(ubar[a], 1e-300)
        riv.append(r)
        lam = (r + 2.0 / (dfcom + 3.0)) / (r + 1.0)
        fmi.append(lam)
        if B[a] <= 0.0:
            df.append(float(dfcom))
        else:
            dold = (m - 1.0) / max(lam * lam, 1e-300)
            dobs = (dfcom + 1.0) / (dfcom + 3.0) * dfcom * (1.0 - lam)
            df.append(dold * dobs / (dold + dobs))

    tq = [_t_quantile(0.975, df[a]) for a in range(p)]
    lo = [qbar[a] - tq[a] * se[a] for a in range(p)]
    hi = [qbar[a] + tq[a] * se[a] for a in range(p)]

    cc = [i for i in range(n) if not any(miss[i])]
    if len(cc) > p and any(ev[i] > 0.5 for i in cc):
        cb, cvv, _cl, _ci, _cc = _cox_breslow([tv[i] for i in cc],
                                              [ev[i] for i in cc],
                                              [obs[i] for i in cc])
        cc_se = [math.sqrt(max(v, 0.0)) for v in cvv]
    else:
        cb, cc_se = [float("nan")] * p, [float("nan")] * p

    return RichResult(payload={
        "estimate": qbar, "coefficients": qbar,
        "hazard_ratio": [math.exp(v) for v in qbar],
        "std_error": se, "total_variance": T,
        "within_variance": ubar, "between_variance": B,
        "ci_lower": lo, "ci_upper": hi,
        "t_quantile": tq, "df": df,
        "relative_increase_variance": riv, "fraction_missing_info": fmi,
        "per_imputation": per,
        "complete_case_coefficients": cb, "complete_case_se": cc_se,
        "n_complete_cases": len(cc),
        "n": n, "p": p, "m": m, "cycles": int(cycles),
        "n_missing": n_missing, "columns_imputed": cols_missing,
        "n_events": n_events, "df_complete": dfcom,
        "method": "multiple imputation by chained equations (norm.nob, "
                  "outcome included in the imputation model), Cox "
                  "proportional hazards with Breslow ties per imputation, "
                  "pooled by Rubin's rules with the Barnard-Rubin degrees "
                  "of freedom (van Buuren 2018 Ch. 3-4; Rubin 1987 Ch. 3)",
        "note": "between_variance is exactly zero when nothing is missing, "
                "so the pooled standard error then equals the "
                "complete-data one; fraction_missing_info is how much of "
                "the final variance came from not knowing the fills",
    })


def cheatsheet():
    return ("sschin: chained_imputation(time, event, X, mi_iter) -> MICE "
            "imputation, per-imputation Cox fits and Rubin-pooled hazard "
            "ratios (van Buuren 2018; Rubin 1987)")
