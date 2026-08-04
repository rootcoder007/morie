# morie.fn -- function file (rootcoder007/morie)
"""Dekkers-Einmahl-de Haan moment estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ev_dedh", "evt_dekkers_einmahl_dehaan"]


def ev_dedh(x, k=None):
    r"""The moment estimator of Dekkers, Einmahl and de Haan (1989),

    .. math:: \hat\xi_M = M_n^{(1)} + 1 - \frac12\left[1 -
              \frac{(M_n^{(1)})^2}{M_n^{(2)}}\right]^{-1},

    with :math:`M_n^{(j)} = \frac1k\sum_{i=0}^{k-1}
    (\log X_{(n-i)} - \log X_{(n-k)})^j` -- the first term is
    exactly the Hill estimator, and the correction extends validity
    to ALL real :math:`\xi`.

    That structure is the estimator's story: for genuinely heavy
    tails the correction term converges to zero and DEdH agrees with
    Hill; for light or bounded tails, where Hill is inconsistent,
    the second moment of the log-excesses rescues it. Both facts are
    tested. Like Hill it needs a positive threshold (logs are
    taken), which is a restriction on the DATA's location, not on
    xi -- shift bounded data above zero and it applies.

    Parameters
    ----------
    x : array-like
        Sample; the top k+1 order statistics must be positive.
    k : int, optional
        Top order statistics used; ``sqrt(n)`` when omitted.

    Returns
    -------
    RichResult
        keys: ``xi``, ``hill_part``, ``correction``, ``M1``, ``M2``,
        ``se``, ``k``, ``threshold``, ``agrees_with_hill_when``,
        ``n``, ``method``.

    References
    ----------
    Dekkers, A. L. M., Einmahl, J. H. J. and de Haan, L. (1989),
    "A moment estimator for the index of an extreme-value
    distribution", *Annals of Statistics* 17:1833-1855, Eq. (1.7).
    """
    from ._evt import top_order

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 10:
        raise ValueError(f"need at least 10 observations, got {n}.")
    kk = int(np.sqrt(n)) if k is None else int(k)
    top = top_order(xv, kk)
    if top[-1] <= 0:
        raise ValueError(
            "the threshold order statistic is not positive; the moment "
            "estimator takes logs, so shift the data above zero first.")
    d = np.log(top[:-1]) - np.log(top[-1])
    M1 = float(np.mean(d))
    M2 = float(np.mean(d ** 2))
    if M2 <= 0:
        raise ValueError("the top order statistics are tied at the "
                         "threshold; no tail information remains.")
    corr = 1.0 - 0.5 / (1.0 - M1 ** 2 / M2)
    xi = M1 + corr
    # asymptotic variance (DEdH Thm 3.1 / de Haan-Ferreira 3.5.4):
    # xi^2 + 1 for xi >= 0; the xi < 0 form is used below it
    if xi >= 0:
        avar = xi ** 2 + 1.0
    else:
        omx = 1.0 - xi
        avar = (omx ** 2 * (1 - 2 * xi)
                * (4 - 8 * (1 - 2 * xi) / (1 - 3 * xi)
                   + (5 - 11 * xi) * (1 - 2 * xi)
                   / ((1 - 3 * xi) * (1 - 4 * xi))))
    return RichResult(payload={
        "xi": xi, "hill_part": M1, "correction": corr,
        "M1": M1, "M2": M2,
        "se": float(np.sqrt(max(avar, 0.0) / kk)),
        "k": kk, "threshold": float(top[-1]),
        "agrees_with_hill_when": "xi > 0: the correction converges to zero "
                                 "and the first term IS the Hill estimator",
        "valid_for": "every real xi; the log still needs positive data, "
                     "a location restriction rather than a tail one",
        "n": int(n),
        "method": "Dekkers-Einmahl-de Haan (1989) moment estimator, Eq. (1.7)"})


def cheatsheet():
    return "evdedh: Hill plus a second-moment correction -- valid for every xi"


#: Catalogue alias for :func:`ev_dedh`.
evt_dekkers_einmahl_dehaan = ev_dedh


# compact alias per ledger/NAMING.md
evdedh = ev_dedh
