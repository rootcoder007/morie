"""Christoffersen conditional-coverage test for VaR exceedances."""

from math import log

from ._richresult import hypothesis_test_result
from . import _stats_core as stats
from .volkupiec import _hit_counts, _lr_uc

__all__ = ["vol_christoffersen_cc"]


def _lr_ind(h):
    """Markov-chain independence LR: first-order chain vs i.i.d."""
    v = [int(x) for x in h.tolist()]
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(v[:-1], v[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    total = n00 + n01 + n10 + n11
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / total
    # log form throughout: an empty transition class contributes 0, and
    # a degenerate chain (pi0 == pi1) gives LR_ind == 0 as it should.
    def _t(k, q):
        return k * log(q) if k else 0.0

    ll_null = _t(n00 + n10, 1.0 - pi) + _t(n01 + n11, pi)
    ll_alt = _t(n00, 1.0 - pi0) + _t(n01, pi0) + _t(n10, 1.0 - pi1) + _t(n11, pi1)
    stat = -2.0 * (ll_null - ll_alt)
    return stat, {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
                  "pi01": pi0, "pi11": pi1, "pi": pi}


def vol_christoffersen_cc(hits, alpha=0.05):
    r"""Christoffersen (1998) conditional-coverage test for a VaR model.

    Splits the null "the exceedance sequence is i.i.d. Bernoulli(alpha)"
    into its two testable halves and adds the statistics:

    .. math:: LR_{cc} = LR_{uc} + LR_{ind} \sim \chi^2_2 .

    ``LR_uc`` is Kupiec's coverage test (:math:`\chi^2_1`). ``LR_ind``
    compares a first-order Markov chain on the indicator against an
    i.i.d. one (:math:`\chi^2_1`), so it fires when breaches *cluster*
    even if their overall rate is right -- the failure mode Kupiec's
    test alone cannot see.

    Parameters
    ----------
    hits : array-like of {0, 1}
        Exceedance indicator in time order. Order matters here.
    alpha : float
        The VaR tail probability the model claims.

    Returns
    -------
    RichResult
        Keys ``statistic`` (LR_cc), ``pvalue``, ``lr_uc``, ``pvalue_uc``,
        ``lr_ind``, ``pvalue_ind``, transition counts ``n00``..``n11``.

    References
    ----------
    Christoffersen, P. F. (1998). Evaluating interval forecasts.
    *International Economic Review*, 39(4), 841-862.
    Definition cross-checked against the reference implementation in
    rugarch (``.LR.cc``, R/rugarch-tests.R).
    """
    h, t, n = _hit_counts(hits)
    uc = _lr_uc(float(alpha), t, n)
    ind, counts = _lr_ind(h)
    cc = uc + ind
    payload = {
        "n_obs": t,
        "n_exceedances": n,
        "lr_uc": float(uc),
        "pvalue_uc": float(stats.chi2.sf(uc, 1)),
        "lr_ind": float(ind),
        "pvalue_ind": float(stats.chi2.sf(ind, 1)),
        "df": 2,
        "alpha": float(alpha),
        "method": "Christoffersen (1998) conditional coverage LR test",
    }
    payload.update(counts)
    return hypothesis_test_result(
        test_name="Christoffersen conditional coverage",
        statistic=float(cc),
        pvalue=float(stats.chi2.sf(cc, 2)),
        extra_summary=[("n_obs", t), ("n_exceedances", n)],
        extra_payload=payload,
    )


def cheatsheet():
    return "volcc: Christoffersen conditional-coverage LR test (LR_uc + LR_ind)"
