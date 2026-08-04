# morie.fn -- function file (rootcoder007/morie)
"""Stratified Cox model."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult
from ._surv import cox_fit, prepare

__all__ = ["cox_stratified"]


def cox_stratified(time, event, X, stratum, ties="efron", max_iter=50, tol=1e-9):
    r"""Fit a Cox model with a separate baseline hazard per stratum.

    The partial likelihood becomes a product over strata,

    .. math::
        L(\beta) = \prod_{s} L_s(\beta),

    with risk sets formed **within** stratum. The coefficient :math:`\beta` is
    shared; the baselines :math:`\lambda_{0s}(t)` are unrestricted and never
    estimated.

    This is the standard remedy when
    :func:`~morie.fn.coxres.cox_schoenfeld_residuals` shows a variable
    violating proportional hazards: stratify on it and the violation is
    absorbed into a free baseline.

    The cost is specific and often overlooked -- **a stratification variable
    has no coefficient, no hazard ratio and no p-value.** You cannot both
    stratify on a variable and estimate its effect. Stratify on the nuisance
    that breaks PH, never on the exposure of interest.

    Strata also cost efficiency: comparisons only ever happen within stratum,
    so many small strata leave little information. A stratum with no events
    contributes nothing at all and is reported.

    Parameters
    ----------
    time, event, X : array-like
        Survival data.
    stratum : array-like
        Stratum label per subject.
    ties : {"efron", "breslow"}
        Tie handling.
    max_iter, tol
        Newton-Raphson controls.

    Returns
    -------
    RichResult
        ``beta``, ``se``, ``z``, ``p_value``, ``hazard_ratio``, ``loglik``,
        ``strata``, ``events_per_stratum``, ``empty_strata``.

    References
    ----------
    Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis of
        Failure Time Data* (2nd ed.). Wiley.
    Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data:
        Extending the Cox Model*. Springer.

    Examples
    --------
    When the baseline genuinely differs by stratum but the effect does not,
    stratifying recovers the shared coefficient.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 1))
    >>> s = rng.integers(0, 3, 600)
    >>> scale = np.array([0.3, 1.0, 4.0])[s]        # very different baselines
    >>> T = rng.exponential(scale / np.exp(X[:, 0] * 0.9))
    >>> C = rng.exponential(4.0, 600)
    >>> t, e = np.minimum(T, C), (T <= C).astype(float)
    >>> r = cox_stratified(t, e, X, s)
    >>> bool(abs(r["beta"][0] - 0.9) < 0.25)
    True

    The stratifier itself gets no coefficient -- there is one per column of X
    and nothing more.

    >>> int(r["beta"].size)
    1
    >>> sorted(r["strata"].tolist())
    [0, 1, 2]

    A stratum with no events contributes nothing and is reported as such.

    >>> s2 = s.copy(); e2 = e.copy(); e2[s2 == 2] = 0.0
    >>> [int(v) for v in cox_stratified(t, e2, X, s2)["empty_strata"]]
    [2]

    >>> cox_stratified(t, e, X, s[:10])
    Traceback (most recent call last):
        ...
    ValueError: stratum has 10 entries but time has 600
    """
    t, e, Xm = prepare(time, event, X)
    st = np.asarray(stratum).ravel()
    if st.size != t.size:
        raise ValueError(f"stratum has {st.size} entries but time has {t.size}")
    levels = np.unique(st)
    n, p = Xm.shape
    beta = np.zeros(p)

    empty = [lv for lv in levels if e[st == lv].sum() == 0]
    converged = False
    it = 0
    ll_total = 0.0
    I_total = np.zeros((p, p))
    for it in range(1, max_iter + 1):
        U = np.zeros(p)
        I_total = np.zeros((p, p))
        ll_total = 0.0
        for lv in levels:
            m = st == lv
            if e[m].sum() == 0:
                continue
            b_s, ll_s, I_s, U_s, _, _ = cox_fit(t[m], e[m], Xm[m], ties=ties,
                                                max_iter=1, tol=tol)
            # One scoring step per stratum at the shared beta.
            _, ll_s, I_s, U_s, _, _ = _score_at(t[m], e[m], Xm[m], beta, ties)
            U += U_s
            I_total += I_s
            ll_total += ll_s
        try:
            step = np.linalg.solve(I_total, U)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(I_total, U, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break

    from ._stats_core import norm

    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(I_total)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    return RichResult(
        title="Stratified Cox model",
        summary_lines=[("n", int(n)), ("strata", int(levels.size)),
                       ("events", int(e.sum())), ("loglik", ll_total)],
        warnings=(["a stratification variable has no coefficient and no hazard "
                   "ratio; stratify on the nuisance, never on the exposure"]
                  + ([f"strata with no events contribute nothing: {empty}"] if empty else [])),
        payload={
            "beta": beta, "se": se, "z": z, "p_value": 2 * norm.sf(np.abs(z)),
            "hazard_ratio": np.exp(beta), "loglik": ll_total,
            "information": I_total, "strata": levels,
            "events_per_stratum": np.array([int(e[st == lv].sum()) for lv in levels]),
            "empty_strata": np.array(empty), "n": int(n),
            "n_iter": it, "converged": converged, "method": "cox_stratified",
        },
    )


def _score_at(t, e, X, beta, ties):
    """Log-likelihood, score and information at a fixed beta (no stepping)."""
    from ._surv import cox_fit as _fit

    saved = np.array(beta, dtype=float, copy=True)
    b, ll, I, U, it, cv = _fit(t, e, X, ties=ties, max_iter=1, tol=np.inf)
    # cox_fit takes one Newton step from zero; recompute at `beta` directly.
    n, p = X.shape
    eta = np.clip(X @ saved, -500, 500)
    w = np.exp(eta)
    U = np.zeros(p)
    I = np.zeros((p, p))
    ll = 0.0
    for ut in np.unique(t[e == 1]):
        at_risk = t >= ut
        died = at_risk & (t == ut) & (e == 1)
        d = int(died.sum())
        if d == 0:
            continue
        wr, Xr = w[at_risk], X[at_risk]
        wd, Xd = w[died], X[died]
        S0r, S1r = wr.sum(), wr @ Xr
        S2r = (wr[:, None] * Xr).T @ Xr
        S0d, S1d = wd.sum(), wd @ Xd
        S2d = (wd[:, None] * Xd).T @ Xd
        ll += eta[died].sum()
        U += Xd.sum(axis=0)
        if ties == "breslow" or d == 1:
            ll -= d * np.log(S0r)
            mu = S1r / S0r
            U -= d * mu
            I += d * (S2r / S0r - np.outer(mu, mu))
        else:
            for l in range(d):
                f = l / d
                S0, S1, S2 = S0r - f * S0d, S1r - f * S1d, S2r - f * S2d
                ll -= np.log(S0)
                mu = S1 / S0
                U -= mu
                I += S2 / S0 - np.outer(mu, mu)
    return saved, float(ll), I, U, 1, True


def cheatsheet():
    return "coxstr: separate baseline per stratum, shared beta; the STRATIFIER gets no hazard ratio"


# compact alias per ledger/NAMING.md
coxstratified = cox_stratified
