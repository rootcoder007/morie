# morie.fn -- function file (rootcoder007/morie)
"""Limited-information maximum likelihood."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_iv_liml"]


def causal_iv_liml(y, X, Z, exog=None, fuller=None, endog=None):
    r"""Limited-information maximum likelihood, as the k-class
    estimator whose ``k`` is the Anderson-Rubin variance ratio.

    Form :math:`\bar Y = [y, X]` and let :math:`M_A = I - P_A`. Then

    .. math:: \kappa = \lambda_{\min}\left[
              (\bar Y' M_{[W,Z]} \bar Y)^{-1}
              (\bar Y' M_W \bar Y)\right],

    with :math:`W` the exogenous regressors alone and :math:`[W,Z]`
    those plus the excluded instruments, and

    .. math:: \hat\beta_{LIML} = \left(X'(I - \kappa M_{[W,Z]})X
              \right)^{-1} X'(I - \kappa M_{[W,Z]})y .

    Two facts make this checkable rather than merely plausible, and
    both are tested. :math:`\kappa \ge 1` always. And when the model
    is EXACTLY identified -- as many excluded instruments as
    endogenous regressors -- :math:`\kappa = 1` and LIML coincides
    with two-stage least squares to machine precision. A LIML
    implementation that does not reproduce 2SLS in the
    just-identified case is wrong, and nothing else about it needs
    checking first.

    Where the two differ is with many or weak instruments. 2SLS is
    biased toward the ordinary least squares estimate, and the bias
    grows with the number of instruments; LIML is approximately
    median-unbiased there, at the cost of heavier tails and, with
    genuinely irrelevant instruments, no finite moments at all.
    ``fuller`` applies Fuller's modification
    :math:`\kappa - a/(n - m)`, which restores finite moments for
    ``a > 0`` at the price of reintroducing some bias; ``a = 1`` is
    the usual choice.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    x : array-like, shape (n, k)
        Regressors, endogenous and exogenous together.
    z : array-like, shape (n, m)
        Full instrument set.
    exog : array-like, optional
        Exogenous regressors alone; the constant when omitted.
    fuller : float, optional
        Fuller's ``a``.
    endog : sequence of int, optional
        Column indices of ``X`` that are endogenous. Detected by
        matching against ``Z`` when omitted -- an included instrument
        appears in both.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se``, ``kappa``, ``kappa_minus_one``,
        ``endogenous_columns``,
        ``just_identified``, ``equals_2sls``, ``fuller_a``,
        ``n_overid_restrictions``, ``n``, ``k``, ``m``, ``method``.

    References
    ----------
    Anderson, T. W. and Rubin, H. (1949), "Estimation of the
    parameters of a single equation in a complete system of
    stochastic equations", *Annals of Mathematical Statistics*
    20:46-63. Fuller, W. A. (1977), *Econometrica* 45:939-953.
    """
    from ._caus_iv import add_intercept, annihilator, k_class

    yv = np.asarray(y, dtype=float).ravel()
    Xm = add_intercept(np.atleast_2d(np.asarray(X, dtype=float)))
    Zm = add_intercept(np.atleast_2d(np.asarray(Z, dtype=float)))
    n, k = Xm.shape
    m = Zm.shape[1]
    if m < k:
        raise ValueError(
            f"the order condition fails: {m} instruments for {k} regressors.")
    # Ybar holds y and the ENDOGENOUS regressors only. Including an
    # exogenous column -- the constant above all -- puts a column in
    # Ybar that M_W annihilates exactly, so B is singular, its
    # smallest eigenvalue is 0, and kappa = 0 turns the k-class
    # estimator back into ordinary least squares. The failure is
    # silent: a finite, plausible-looking, entirely wrong number.
    if endog is None:
        # an INCLUDED instrument appears as a column of Z; whatever is
        # left over in X is the endogenous part
        is_exog = np.zeros(k, dtype=bool)
        for j in range(k):
            xj = Xm[:, j]
            nj = np.linalg.norm(xj)
            if nj == 0:
                is_exog[j] = True
                continue
            for c in range(m):
                if np.linalg.norm(xj - Zm[:, c]) <= 1e-10 * max(nj, 1.0):
                    is_exog[j] = True
                    break
    else:
        idx = np.atleast_1d(np.asarray(endog, dtype=int)).ravel()
        if np.any((idx < 0) | (idx >= k)):
            raise ValueError(f"endog indices must lie in 0..{k - 1}.")
        is_exog = np.ones(k, dtype=bool)
        is_exog[idx] = False
    if np.all(is_exog):
        raise ValueError(
            "no endogenous regressor was identified: every column of X also "
            "appears in Z, so there is nothing to instrument and LIML "
            "reduces to least squares. Pass endog explicitly if the "
            "detection is wrong.")
    W = (add_intercept(Xm[:, is_exog]) if exog is None
         else add_intercept(np.atleast_2d(np.asarray(exog, dtype=float))))
    Ybar = np.column_stack([yv, Xm[:, ~is_exog]])
    A = Ybar.T @ annihilator(Zm, Ybar)
    B = Ybar.T @ annihilator(W, Ybar)
    ev = np.linalg.eigvals(np.linalg.pinv(A) @ B)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        raise ValueError("the Anderson-Rubin ratio has no finite eigenvalue.")
    kappa = float(np.min(ev.real))
    nres = m - k
    if fuller is not None:
        a = float(fuller)
        if a < 0:
            raise ValueError(f"Fuller's a must be non-negative, got {a}.")
        kappa = kappa - a / (n - m)
    beta = k_class(yv, Xm, Zm, kappa)
    u = yv - Xm @ beta
    MzX = annihilator(Zm, Xm)
    A2 = Xm.T @ Xm - kappa * (Xm.T @ MzX)
    bread = np.linalg.pinv(A2)
    Xt = Xm - kappa * MzX
    V = bread @ (Xt.T @ (Xt * (u ** 2)[:, None])) @ bread
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return RichResult(payload={
        "beta": beta, "se": se, "residuals": u,
        "kappa": kappa, "kappa_minus_one": kappa - 1.0,
        "just_identified": bool(nres == 0),
        "endogenous_columns": np.flatnonzero(~is_exog),
        "equals_2sls": bool(abs(kappa - 1.0) < 1e-9),
        "fuller_a": None if fuller is None else float(fuller),
        "n_overid_restrictions": int(nres),
        "kappa_fact": "kappa >= 1 always, and equals 1 exactly when the "
                      "model is just identified, where LIML IS 2SLS",
        "versus_2sls": "2SLS is biased toward OLS and the bias grows with "
                       "the instrument count; LIML is approximately "
                       "median-unbiased there but has heavier tails and, "
                       "with irrelevant instruments, no finite moments",
        "n": int(n), "k": int(k), "m": int(m),
        "method": "LIML as the k-class estimator with k = the Anderson-Rubin ratio"})


def cheatsheet():
    return "causivlim: kappa >= 1, and kappa == 1 exactly when just identified -- then LIML IS 2SLS"


# compact alias per ledger/NAMING.md
causalivliml = causal_iv_liml
