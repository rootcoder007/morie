# morie.fn -- function file (rootcoder007/morie)
"""WAIC with its effective parameter count."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["waic_diagnostic"]


def waic_diagnostic(log_lik):
    """Predictive criterion computed from the posterior, pointwise.

    AIC and DIC both need a point estimate of the parameters; WAIC does
    not, which is what makes it usable for singular models where the
    likelihood is not asymptotically normal around a maximum.  The
    effective parameter count falls out as the posterior variance of
    the pointwise log likelihood, and when that variance is large for
    some observation the criterion is warning that the posterior is
    being stretched by that point.

    Formula: ``lppd = sum_i log mean_s exp(ll_is)``,
    ``p_WAIC = sum_i Var_s(ll_is)``, ``WAIC = -2(lppd - p_WAIC)``.

    Parameters
    ----------
    log_lik : array-like, shape (S, n)
        Pointwise log likelihood, posterior draws by observations.

    Returns
    -------
    RichResult
        ``estimate`` (WAIC), ``lppd``, ``p_waic``, ``elpd``,
        ``n_high_var`` (observations whose variance exceeds 0.4, the
        usual warning threshold), ``S``, ``n``.

    References
    ----------
    Watanabe, S. (2010).  Asymptotic equivalence of Bayes cross
    validation and widely applicable information criterion in singular
    learning theory.  Journal of Machine Learning Research
    11:3571-3594.  The 0.4 variance flag is Vehtari, A., Gelman, A. &
    Gabry, J. (2017), Statistics and Computing 27:1413-1432.
    """
    L = C.mat(log_lik)
    Sn = len(L)
    n = len(L[0])
    lppd = 0.0
    pw = 0.0
    high = 0
    for i in range(n):
        col = [L[s][i] for s in range(Sn)]
        m = max(col)
        lppd += m + math.log(sum(math.exp(v - m) for v in col) / Sn)
        mu = sum(col) / Sn
        v = sum((t - mu) ** 2 for t in col) / (Sn - 1)
        pw += v
        if v > 0.4:
            high += 1
    return RichResult(payload={
        "estimate": -2.0 * (lppd - pw), "lppd": lppd, "p_waic": pw,
        "elpd": lppd - pw, "n_high_var": high, "S": Sn, "n": n,
        "method": "WAIC with effective parameter count"})


waicdiagnostic = waic_diagnostic


def cheatsheet():
    return "waicd: WAIC with its effective parameter count."
