# morie.fn -- function file (rootcoder007/morie)
"""Likelihood ratio test."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lrtest", "wasserman_lrt"]


def lrtest(loglik_full, loglik_null, df):
    """Likelihood ratio test from two maximised log-likelihoods.

    The degrees of freedom are the DIFFERENCE IN DIMENSION, not the
    number of parameters in either model -- the single most common way
    to get this test wrong.  A negative statistic means the "null"
    model fitted better than the unrestricted one, which is impossible
    if the models are genuinely nested, so it is raised rather than
    clamped to zero.

    Formula: lambda = 2 (l_full - l_null);  p = P(chi^2_df > lambda)

    Parameters
    ----------
    loglik_full : float
        Maximised log-likelihood over the whole parameter space.
    loglik_null : float
        Maximised log-likelihood over the null subspace.
    df : int
        dim(Theta) - dim(Theta_0), at least 1.

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``df``, ``loglik_full``,
        ``loglik_null``.

    References
    ----------
    Wasserman (2004), All of Statistics, Definition 10.21 -- the
    statistic is lambda = 2 log(L(theta_hat)/L(theta_hat_0)) -- and
    Theorem 10.22, under which lambda converges to chi^2 with
    r - q degrees of freedom, "the dimension of Theta minus the
    dimension of Theta_0", with p-value P(chi^2_{r-q} > lambda).
    Fetched as the full text of the book.
    """
    lf = float(loglik_full)
    ln = float(loglik_null)
    df = int(df)
    if df < 1:
        raise ValueError("df must be at least 1")
    lam = 2.0 * (lf - ln)
    if lam < 0:
        raise ValueError(
            "the unrestricted log-likelihood is below the restricted one; "
            "the models are not nested or one did not converge")
    return RichResult(payload={
        "statistic": lam, "p_value": 1.0 - C.pchisq(lam, df),
        "df": float(df), "loglik_full": lf, "loglik_null": ln,
        "method": "Likelihood ratio test, Wasserman Theorem 10.22"})


wasserman_lrt = lrtest


def cheatsheet():
    return "wsmlrt: lambda = 2(l_full - l_null) ~ chi^2_df"
