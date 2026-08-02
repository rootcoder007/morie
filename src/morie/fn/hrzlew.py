# morie.fn -- function file (rootcoder007/morie)
"""Lewbel heteroskedastic binary response estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["horowitz_lewbel_estimator"]


def horowitz_lewbel_estimator(x, y, z, bandwidth=None, instruments=None,
                              density="nonparametric"):
    r"""Lewbel's special-regressor estimator for a heteroskedastic
    binary-response model (Horowitz Sec. 4.5):

    .. math:: Y = \mathbf 1\{V + X'\beta + \varepsilon > 0\},

    where :math:`V` is the SPECIAL REGRESSOR -- continuously
    distributed with large support, entering additively, and with a
    coefficient KNOWN to be 1 (that is the normalisation, not an
    assumption about magnitude). Writing :math:`V = S'b + U` with
    :math:`U \perp (S, \varepsilon)`, the estimator constructs

    .. math:: T = \frac{Y - \mathbf 1\{V \ge 0\}}{f(U)}

    and recovers beta from a LINEAR regression of T on X -- ordinary
    least squares when X is exogenous, two-stage least squares on
    instruments Z when it is not. The point of the construction is
    that :math:`T = X'\beta + \tilde\varepsilon` with
    :math:`E(Z\tilde\varepsilon) = 0`, so a discrete-choice problem
    collapses to a linear one.

    Why the chapter reaches for this at all: maximum score and
    smoothed maximum score are consistent under weak assumptions but
    do NOT identify :math:`P(Y = 1|X = x)`, and converge more slowly
    than :math:`n^{-1/2}`. Lewbel's estimator buys back both --
    root-n consistency and the choice probabilities -- at the price
    of needing a special regressor, and it still permits
    heteroskedasticity in :math:`\varepsilon` of unknown form.

    The dividing density is what makes it fragile: :math:`\hat f(U)`
    appears in a DENOMINATOR, so observations in the tails of U carry
    enormous weight, and the source itself notes that outliers may
    need discarding at the regression step. The smallest fitted
    density and the resulting weight range are therefore returned
    rather than left implicit.

    Parameters
    ----------
    x : array-like, shape (n, d)
        Regressors, excluding the special regressor. A constant is
        added if none is present.
    y : array-like of {0, 1}, shape (n,)
        Binary response.
    z : array-like, shape (n,)
        The special regressor V.
    bandwidth : float, optional
        Bandwidth for the nonparametric density of U; Silverman's
        rule otherwise. Ignored when ``density="normal"``.
    instruments : array-like, optional
        Instruments for endogenous columns of X. Ordinary least
        squares is used when omitted.
    density : {"nonparametric", "normal"}
        How ``f(U)`` is estimated. ``"normal"`` is the parametric
        shortcut of Estimator 1, step 2.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se``, ``coefficient_on_V`` (1.0, the
        normalisation), ``min_density``, ``max_weight``,
        ``root_n_consistent`` (True),
        ``heteroskedasticity_allowed`` (True),
        ``identifies_choice_probabilities`` (True), ``bandwidth``,
        ``endogenous`` (whether instruments were used), ``n``, ``d``,
        ``method``.
    References
    ----------
    Horowitz, J. L. *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Sec. 4.5 (other estimators for
    heteroskedastic binary-response models), which describes Lewbel
    (2000) but prints no formula.

    The formula implemented here is taken from the primary source
    rather than reconstructed: Dong, Y. and Lewbel, A., *Simple
    Estimators for Binary Choice Models With Endogenous Regressors*,
    Corollary 1 and Estimator 1 -- ``T = [D - I(V >= 0)] / f(U)``
    with ``U`` the residual of V on S, then 2SLS of T on X using
    instruments Z. Note the indicator is ``I(V >= 0)``; at least one
    secondary description of this estimator states ``I(V < 0)``,
    which changes the estimand.
    """
    from . import _stats_core as stats

    from ._horowitz import kernel, silverman_bw

    X = np.atleast_2d(np.asarray(x, dtype=float))
    yv = np.asarray(y, dtype=float).ravel()
    V = np.asarray(z, dtype=float).ravel()
    if X.shape[0] != yv.size:
        X = X.T
    if X.shape[0] != yv.size:
        raise ValueError("x must have one row per entry of y.")
    if V.size != yv.size:
        raise ValueError(
            f"z must have one entry per row of x, got {V.size} for {yv.size}.")
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("y must be binary 0/1.")
    if density not in ("nonparametric", "normal"):
        raise ValueError("density must be 'nonparametric' or 'normal'.")
    n = yv.size
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")

    # a constant belongs in X and in the instrument set
    has_const = np.any(np.all(np.isclose(X, X[0, :]), axis=0))
    Xd = X if has_const else np.column_stack([np.ones(n), X])
    d = Xd.shape[1]

    Zi = None if instruments is None else np.atleast_2d(
        np.asarray(instruments, dtype=float))
    if Zi is not None:
        if Zi.shape[0] != n:
            Zi = Zi.T
        if Zi.shape[0] != n:
            raise ValueError("instruments must have one row per observation.")
        Zi = np.column_stack([np.ones(n), Zi]) if not np.any(
            np.all(np.isclose(Zi, Zi[0, :]), axis=0)) else Zi
        if Zi.shape[1] < d:
            raise ValueError(
                f"need at least {d} instruments for {d} regressors, "
                f"got {Zi.shape[1]}.")

    # Step 1: demean V, then take residuals of V on S = all regressors
    # and instruments (everything except V itself)
    Vc = V - V.mean()
    S = Xd if Zi is None else np.column_stack([Xd, Zi[:, 1:]])
    coef, *_ = np.linalg.lstsq(S, Vc, rcond=None)
    U = Vc - S @ coef

    # Step 2: f(U), nonparametrically or via the normal shortcut
    if density == "normal":
        sd = float(np.sqrt(np.mean(U**2)))
        if sd <= 0:
            raise ValueError("the special regressor is fully explained by S; "
                             "U has zero variance and f(U) is undefined.")
        fhat = stats.norm.pdf(U / sd) / sd
        hh = None
    else:
        hh = silverman_bw(U) if bandwidth is None else float(bandwidth)
        if hh <= 0:
            raise ValueError(f"bandwidth must be positive, got {hh}.")
        fhat = kernel((U[:, None] - U[None, :]) / hh).sum(axis=1) / (n * hh)
    if np.any(fhat <= 0):
        raise ValueError("the fitted density of U vanishes at some "
                         "observation; T would be undefined there.")

    # Step 3: T = [Y - I(V >= 0)] / f(U)
    T = (yv - (V >= 0.0).astype(float)) / fhat

    # Step 4: linear regression of T on X, by 2SLS when instruments
    # are supplied
    if Zi is None:
        A = Xd
    else:
        pz, *_ = np.linalg.lstsq(Zi, Xd, rcond=None)
        A = Zi @ pz
    beta, *_ = np.linalg.lstsq(A, T, rcond=None)
    resid = T - Xd @ beta
    dof = max(n - d, 1)
    xtx_inv = np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(xtx_inv) * float(resid @ resid) / dof)

    return RichResult(payload={
        "beta": beta, "se": se, "coefficient_on_V": 1.0,
        "min_density": float(fhat.min()),
        "max_weight": float((1.0 / fhat).max()),
        "root_n_consistent": True,
        "heteroskedasticity_allowed": True,
        "identifies_choice_probabilities": True,
        "bandwidth": hh, "endogenous": Zi is not None,
        "n": int(n), "d": int(d),
        "method": "Lewbel special regressor: T = [Y - I(V >= 0)] / f(U), then a linear regression"})


def cheatsheet():
    return "hrzlew: f(U) sits in a DENOMINATOR -- tail observations dominate, so watch max_weight"
