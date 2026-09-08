# morie.fn -- function file (rootcoder007/morie)
"""Two-stage least squares."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_iv_2sls"]


def causal_iv_2sls(y, X, Z, cluster=None):
    r"""Two-stage least squares,

    .. math:: \hat\beta_{2SLS} = (X'P_Z X)^{-1}X'P_Z y,
              \qquad P_Z = Z(Z'Z)^{-1}Z' .

    Identification needs at least as many instruments as regressors
    (the order condition) AND :math:`\mathrm{rank}(Z'X) = \dim\beta`
    (the rank condition). The order condition is arithmetic and is
    checked here; the rank condition is checked numerically through
    the conditioning of :math:`Z'X`, which is the honest version --
    rank is not a decidable property of floating-point data, only a
    matter of degree, and the degree is what ``first_stage_F``
    reports.

    The standard errors are the heteroskedasticity-robust sandwich
    :math:`(\hat X'\hat X)^{-1}\hat X'\,\mathrm{diag}(u^2)\,\hat
    X(\hat X'\hat X)^{-1}` with :math:`\hat X = P_Z X`, and the
    residuals :math:`u = y - X\hat\beta` use the ORIGINAL ``X``, not
    the fitted one. Using the first-stage fitted values there is a
    common and silent error: it produces a smaller number that is not
    a standard error of anything.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    x : array-like, shape (n, k)
        Regressors, endogenous and exogenous together. Include a
        constant column or one is added.
    z : array-like, shape (n, m)
        Instruments, m >= k. Exogenous regressors instrument for
        themselves and belong in both.
    cluster : array-like, optional
        Cluster identifiers for a clustered sandwich.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se``, ``t``, ``residuals``, ``fitted``,
        ``first_stage_F``, ``order_condition``, ``overidentified``,
        ``n_overid_restrictions``, ``sargan``, ``sargan_p``,
        ``vcov_type``, ``n``, ``k``, ``m``, ``method``.

    References
    ----------
    Wooldridge, J. M. (2010), *Econometric Analysis of Cross Section
    and Panel Data*, 2nd ed., MIT Press, Ch. 5. Sargan, J. D. (1958),
    *Econometrica* 26:393-415, for the overidentification test.
    """
    from . import _stats_core as stats

    from ._caus_iv import add_intercept, annihilator, first_stage_f, projection

    yv = np.asarray(y, dtype=float).ravel()
    Xm = add_intercept(np.atleast_2d(np.asarray(X, dtype=float)))
    Zm = add_intercept(np.atleast_2d(np.asarray(Z, dtype=float)))
    if Xm.shape[0] != yv.size:
        Xm = add_intercept(np.atleast_2d(np.asarray(X, dtype=float)).T)
    n, k = Xm.shape
    m = Zm.shape[1]
    if Zm.shape[0] != n:
        raise ValueError(
            f"Z has {Zm.shape[0]} rows for {n} observations.")
    if m < k:
        raise ValueError(
            f"the order condition fails: {m} instruments (including the "
            f"constant) for {k} regressors. 2SLS is not identified.")
    Xhat = projection(Zm, Xm)
    XtX = Xhat.T @ Xhat
    if np.linalg.matrix_rank(XtX) < k:
        raise ValueError(
            "the rank condition fails: the projected regressors are "
            "collinear, so Z carries no independent variation for at least "
            "one endogenous regressor.")
    beta = np.linalg.lstsq(XtX, Xhat.T @ yv, rcond=None)[0]
    # residuals use the ORIGINAL X, not the projection
    u = yv - Xm @ beta
    bread = np.linalg.pinv(XtX)
    if cluster is None:
        meat = Xhat.T @ (Xhat * (u ** 2)[:, None])
        vt = "heteroskedasticity-robust (HC0)"
    else:
        cl = np.asarray(cluster).ravel()
        if cl.size != n:
            raise ValueError(f"cluster has {cl.size} entries for {n} rows.")
        meat = np.zeros((k, k))
        for g in np.unique(cl):
            s = cl == g
            v = Xhat[s].T @ u[s]
            meat += np.outer(v, v)
        vt = f"cluster-robust ({np.unique(cl).size} clusters)"
    V = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    # Sargan: n R^2 from regressing the residuals on all instruments
    nres = m - k
    sargan = sargan_p = None
    if nres > 0:
        r2 = 1.0 - float(np.sum(annihilator(Zm, u) ** 2)) / float(u @ u)
        sargan = float(n * r2)
        sargan_p = float(stats.chi2.sf(sargan, nres))
    excl = Zm[:, ~np.all(np.isclose(Zm, 1.0), axis=0)]
    return RichResult(payload={
        "beta": beta, "se": se,
        "t": np.divide(beta, se, out=np.full(k, np.nan), where=se > 0),
        "residuals": u, "fitted": Xm @ beta,
        "first_stage_F": first_stage_f(Xm[:, -1], excl),
        "order_condition": True,
        "overidentified": bool(nres > 0), "n_overid_restrictions": int(nres),
        "sargan": sargan, "sargan_p": sargan_p,
        "vcov_type": vt,
        "residual_note": "residuals are y - X beta with the ORIGINAL X; "
                         "using the first-stage fitted values gives a "
                         "smaller number that is not a standard error",
        "n": int(n), "k": int(k), "m": int(m),
        "method": "Two-stage least squares, beta = (X'P_Z X)^-1 X'P_Z y"})


def cheatsheet():
    return "causiv2sls: residuals use the ORIGINAL X, never the first-stage fit"


# compact alias per ledger/NAMING.md
causaliv2sls = causal_iv_2sls
