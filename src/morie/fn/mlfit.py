# morie.fn -- function file (rootcoder007/morie)
"""ML log-likelihood of a linear mixed model (alias of :mod:`lmmll`)."""

from .lmmll import lmm_loglik

from ._richresult import RichResult

__all__ = ["ml_loglik", "mlloglik"]


def ml_loglik(y, X, V):
    """Gaussian ML log-likelihood of a linear mixed model at a given V.

    This module is an ALIAS.  The likelihood is implemented once, in
    ``lmmll.lmm_loglik``; this entry point is the restriction of it to
    the case where the marginal variance is supplied directly and the
    objective is ML rather than REML.  No second copy exists.

        l(beta, V; y) = -1/2 [ n log(2 pi) + log|V|
                               + (y - X beta)' V^-1 (y - X beta) ]

    with ``beta`` profiled out at its GLS value
    ``(X' V^-1 X)^-1 X' V^-1 y``, which is the maximiser for any fixed
    ``V``, so what is returned is the profile log-likelihood in ``V``.
    ``V`` must be positive definite; an inadmissible set of variance
    components is refused rather than returned as a complex number.

    Parameters
    ----------
    y : array-like, shape (n,)
        Response.
    X : array-like, shape (n, p)
        Fixed-effects design.
    V : array-like, shape (n, n)
        Marginal variance ``Z D Z' + R``.

    Returns
    -------
    RichResult
        ``estimate`` (log-likelihood), ``loglik``, ``neg2loglik``,
        ``logdet_V``, ``quadratic_form``, ``aic``, ``bic``, ``n``, ``p``.

    References
    ----------
    Hartley, H. O. and Rao, J. N. K. (1967), "Maximum-likelihood
    estimation for the mixed analysis of variance model", Biometrika
    54(1-2), 93-108, doi:10.1093/biomet/54.1-2.93, which is where the
    mixed-model ML objective and its profiling over beta originate.
    """
    r = lmm_loglik(y, X, V=V, reml=False)
    return RichResult(payload={
        "estimate": r["loglik"], "loglik": r["loglik"],
        "neg2loglik": r["neg2loglik"], "logdet_V": r["logdet_V"],
        "quadratic_form": r["quadratic_form"], "aic": r["aic"],
        "bic": r["bic"], "n": r["n"], "p": r["p"],
        "method": "ML log-likelihood of a linear mixed model"})


mlloglik = ml_loglik


def cheatsheet():
    return "mlfit: ML log-likelihood of a linear mixed model (alias of lmmll)"
