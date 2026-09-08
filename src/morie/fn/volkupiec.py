"""Kupiec unconditional-coverage likelihood-ratio test for VaR exceedances."""

from math import log

from ._richresult import hypothesis_test_result
from . import _array_core as np
from . import _stats_core as stats

__all__ = ["vol_kupiec_var_test"]


def _hit_counts(hits):
    h = np.asarray(hits, dtype=float).ravel()
    t = int(h.size)
    if t < 2:
        raise ValueError("need at least 2 observations in the hit sequence.")
    for v in h.tolist():
        if v not in (0.0, 1.0):
            raise ValueError("hits must be a 0/1 exceedance indicator sequence.")
    return h, t, int(sum(1 for v in h.tolist() if v == 1.0))


def _lr_uc(p, t, n):
    """-2 log( L(p) / L(n/t) ) for a Bernoulli(p) exceedance sequence."""
    if not 0.0 < p < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    phat = n / t
    # both likelihoods are written in log form so the 0^0 = 1 corner
    # (n = 0 or n = t) contributes exactly zero rather than log(0).
    ll_null = (t - n) * log(1.0 - p) + (n * log(p) if n else 0.0)
    ll_alt = ((t - n) * log(1.0 - phat) if n < t else 0.0) + (n * log(phat) if n else 0.0)
    return -2.0 * (ll_null - ll_alt)


def vol_kupiec_var_test(hits, alpha=0.05):
    r"""Kupiec (1995) proportion-of-failures test for a VaR model.

    Under the null the exceedance indicator is i.i.d. Bernoulli(alpha),
    so the likelihood ratio against the unrestricted Bernoulli MLE

    .. math::

       LR_{uc} = -2\log\frac{(1-p)^{T-N}p^{N}}
                            {(1-N/T)^{T-N}(N/T)^{N}}

    is asymptotically :math:`\chi^2_1`. This is a test of *coverage
    only*: a model that puts every one of its exceedances in a single
    cluster passes it. Pair it with the independence half --
    :func:`vol_christoffersen_cc` -- before concluding a model is sound.

    Parameters
    ----------
    hits : array-like of {0, 1}
        Exceedance indicator, 1 when the realised loss breached VaR.
    alpha : float
        The VaR tail probability the model claims, e.g. 0.05.

    Returns
    -------
    RichResult
        Keys ``statistic`` (LR_uc), ``pvalue``, ``n_exceedances``,
        ``n_obs``, ``expected_exceedances``, ``rate``.

    References
    ----------
    Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
    measurement models. *Journal of Derivatives*, 3(2), 73-84.
    Definition cross-checked against the reference implementation in
    rugarch (``.LR.uc``, R/rugarch-tests.R).
    """
    _, t, n = _hit_counts(hits)
    stat = _lr_uc(float(alpha), t, n)
    p = float(stats.chi2.sf(stat, 1))
    return hypothesis_test_result(
        test_name="Kupiec unconditional coverage",
        statistic=float(stat),
        pvalue=p,
        extra_summary=[("n_obs", t), ("n_exceedances", n)],
        extra_payload={
            "n_obs": t,
            "n_exceedances": n,
            "expected_exceedances": float(alpha) * t,
            "rate": n / t,
            "df": 1,
            "alpha": float(alpha),
            "method": "Kupiec (1995) unconditional coverage LR test",
        },
    )


def cheatsheet():
    return "volkupiec: Kupiec unconditional-coverage LR test for VaR exceedances"
