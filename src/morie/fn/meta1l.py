# SPDX-License-Identifier: AGPL-3.0-or-later
"""S-, T-, X- and R-metalearners for the CATE with OLS base learners."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["meta1l", "metalearner_ensemble"]


def _ols(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


def _design(X):
    n = X.shape[0]
    return np.concatenate([np.ones((n, 1)), X], axis=1)


def meta1l(y, w, X, ps=None):
    """
    The four standard metalearners of the conditional average
    treatment effect -- S, T, X (Kunzel et al. 2019) and R (Nie and
    Wager 2021) -- with ordinary least squares as every base learner,
    evaluated at the sample covariates.

    T-learner (Kunzel et al., Eq. 3): fit mu0 on controls and mu1 on
    treated, tau_T(x) = mu1(x) - mu0(x). S-learner (Eq. 4): fit one
    regression mu(x, w) with the treatment indicator as a plain
    feature (no interactions), tau_S(x) = mu(x, 1) - mu(x, 0), which
    under OLS is the constant coefficient on w. X-learner
    (Eqs. 5-9): impute D1 = Y1 - mu0(X1) on the treated and
    D0 = mu1(X0) - Y0 on the controls, regress each on x to get tau1
    and tau0, and combine tau_X(x) = g(x) tau0(x) + (1 - g(x)) tau1(x)
    with g the propensity score (their Remark 1 recommendation);
    a constant g = mean(w) is used when ``ps`` is not supplied.
    R-learner (Nie and Wager, Eq. 4): minimise the R-loss
    sum((Y - m(X) - (W - e(X)) tau(x))^2) over linear tau, i.e. OLS of
    the outcome residual on the treatment residual times the design;
    m and e are OLS fits of y and w on x (Robinson residualization).

    On a saturated linear data-generating process all four reduce to
    differences of OLS fits and agree exactly.

    Parameters
    ----------
    y : array-like
        Outcome, length n.
    w : array-like
        Binary treatment, 0/1.
    X : array-like, shape (n, p)
        Covariates.
    ps : array-like, optional
        Propensity scores e(x) used as the X-learner weight g(x) and
        as the R-learner treatment expectation. Defaults to OLS of w
        on X for the R-learner residual and mean(w) for g.

    Returns
    -------
    result : RichResult
        Keys: estimate (dict of the four ATE estimates, the mean CATE
        per learner), cate_s, cate_t, cate_x, cate_r (per-unit CATE
        vectors), coef_r (R-learner tau coefficients), n, n_treat.

    References
    ----------
    Kunzel, S. R., Sekhon, J. S., Bickel, P. J. and Yu, B. (2019),
    "Metalearners for estimating heterogeneous treatment effects
    using machine learning", PNAS 116(10), 4156-4165,
    doi:10.1073/pnas.1804597116, Eqs. 3-9 and Remark 1. Local copy:
    fetched-wave3/kunzel-sekhon-bickel-yu-2019-metalearners-heterogeneous-treatment-effects-PNAS116.pdf
    Nie, X. and Wager, S. (2021), "Quasi-oracle estimation of
    heterogeneous treatment effects", Biometrika 108(2), 299-319,
    doi:10.1093/biomet/asaa076, Eq. 4 (R-loss). Local copy:
    fetched-wave3/nie-wager-2021-quasi-oracle-heterogeneous-treatment-effects-Biometrika108.pdf
    Lead reference: Curth, A. and van der Schaar, M. (2021),
    "Nonparametric estimation of heterogeneous treatment effects:
    From theory to learning algorithms", AISTATS 130 (arXiv:
    2101.10943), which surveys the same learner family. Local copy:
    fetched-wave3/curth-vanderschaar-2021-nonparametric-hte-theory-to-learning-AISTATS.pdf
    """
    yv = np.asarray(y, dtype=float)
    wv = np.asarray(w, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape((-1, 1))
    n = len(yv)
    if Xa.shape[0] != n or len(wv) != n:
        raise ValueError("y, w, X must have matching first dimension")
    for v in wv:
        if float(v) not in (0.0, 1.0):
            raise ValueError("w must be binary 0/1")
    i1 = [i for i in range(n) if wv[i] == 1.0]
    i0 = [i for i in range(n) if wv[i] == 0.0]
    p = Xa.shape[1]
    if len(i1) <= p + 1 or len(i0) <= p + 1:
        raise ValueError("need more than p + 1 units in each arm")
    D = _design(Xa)

    # T-learner
    D1 = np.stack([D[i] for i in i1], axis=0)
    D0 = np.stack([D[i] for i in i0], axis=0)
    y1 = np.asarray([yv[i] for i in i1])
    y0 = np.asarray([yv[i] for i in i0])
    b1 = _ols(D1, y1)
    b0 = _ols(D0, y0)
    cate_t = D @ b1 - D @ b0

    # S-learner: treatment as a plain feature
    Ds = np.concatenate([D, wv.reshape((-1, 1))], axis=1)
    bs = _ols(Ds, yv)
    cate_s = np.full(n, float(bs[p + 1]))

    # X-learner
    d1 = y1 - D1 @ b0
    d0 = D0 @ b1 - y0
    t1 = _ols(D1, d1)
    t0 = _ols(D0, d0)
    if ps is not None:
        g = np.asarray(ps, dtype=float)
    else:
        g = np.full(n, float(np.mean(wv)))
    cate_x = g * (D @ t0) + (1.0 - g) * (D @ t1)

    # R-learner: Robinson residualization + OLS on the R-loss
    m_hat = D @ _ols(D, yv)
    if ps is not None:
        e_hat = np.asarray(ps, dtype=float)
    else:
        e_hat = D @ _ols(D, wv)
    ry = yv - m_hat
    rw = wv - e_hat
    Dr = np.stack([D[i] * rw[i] for i in range(n)], axis=0)
    br = _ols(Dr, ry)
    cate_r = D @ br

    return RichResult(payload={
        "estimate": {
            "s": float(np.mean(cate_s)), "t": float(np.mean(cate_t)),
            "x": float(np.mean(cate_x)), "r": float(np.mean(cate_r)),
        },
        "cate_s": cate_s, "cate_t": cate_t,
        "cate_x": cate_x, "cate_r": cate_r,
        "coef_r": br,
        "n": n, "n_treat": len(i1),
        "method": "S/T/X (Kunzel et al. 2019) + R (Nie-Wager 2021) metalearners, OLS base learners",
    })


metalearner_ensemble = meta1l


def cheatsheet():
    return "meta1l(y, w, X, ps) -> S/T/X/R metalearner CATEs with OLS base learners."

# public names resolved by fn/_lazy_map.json
meta_learner_ensemble = meta1l
