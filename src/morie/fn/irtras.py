"""Andrich rating scale model across items with shared category thresholds."""

from __future__ import annotations

from ._irtcore import broadcast, cat_moments, rsm_probs, seq_
from ._richresult import RichResult

__all__ = ["rating_scale_model"]


def rating_scale_model(theta, b, tau):
    r"""Rating scale model for a set of items sharing one threshold set.

    Every item :math:`i` has its own location :math:`b_i` but all items
    share the same :math:`m` category thresholds
    :math:`\tau_1, \ldots, \tau_m` -- that shared threshold structure is
    what distinguishes the rating scale model from the partial credit
    model, in which each item gets its own thresholds. For person
    :math:`v` and item :math:`i`,

    .. math::
        P(X_{vi} = h) \propto \exp\{h(\theta_v - b_i) - \textstyle\sum_{j\le h}\tau_j\}.

    The previous body was a placeholder: it averaged a leading ``X``
    argument and used ``ncats`` for nothing. Both are gone.

    Parameters
    ----------
    theta : float or array-like
        Person abilities, length ``n``.
    b : float or array-like
        Item locations, length ``k`` (or 1, recycled).
    tau : array-like
        The ``m`` thresholds shared by all items; ``m + 1`` categories.

    Returns
    -------
    RichResult
        ``p`` (a list of ``k`` matrices, item by item, each ``n`` rows by
        ``m + 1`` columns), ``expected`` (``n`` by ``k`` expected category
        scores), ``info`` (``n`` by ``k`` item informations), ``test_expected``
        and ``test_info`` (row sums over items), ``theta``, ``b``, ``tau``,
        ``ncat``, ``n``, ``k``, ``method``.

    Notes
    -----
    Item information for this model is exactly the variance of the category
    score, because the linear predictor is :math:`h\theta` plus terms free
    of :math:`\theta`; test information is the sum over items by local
    independence.

    References
    ----------
    Andrich, D. (1978). A rating formulation for ordered response
    categories. *Psychometrika*, 43(4), 561-573. doi:10.1007/BF02293814

    Mair, P. & Hatzinger, R. (2007). Extended Rasch modeling: the eRm
    package for the application of IRT models in R. *Journal of Statistical
    Software*, 20(9), eq. (5), p. 4.
    """
    th = [float(v) for v in seq_(theta)]
    n = len(th)
    if n == 0:
        raise ValueError("theta is empty.")
    bs = seq_(b)
    k = len(bs)
    if k == 0:
        raise ValueError("b is empty.")
    bv = broadcast(bs, k, "b")
    tv = [float(v) for v in seq_(tau)]
    if len(tv) == 0:
        raise ValueError("tau is empty; a rating scale needs at least one threshold.")
    scores = list(range(len(tv) + 1))

    p = []
    expected = [[0.0] * k for _ in range(n)]
    info = [[0.0] * k for _ in range(n)]
    for i in range(k):
        rows = []
        for v in range(n):
            pr, _ = rsm_probs(th[v], bv[i], tv)
            mu, var = cat_moments(pr, scores)
            rows.append(pr)
            expected[v][i] = mu
            info[v][i] = var
        p.append(rows)

    return RichResult(
        payload={
            "p": p,
            "expected": expected,
            "info": info,
            "test_expected": [sum(r) for r in expected],
            "test_info": [sum(r) for r in info],
            "theta": th,
            "b": bv,
            "tau": tv,
            "ncat": len(tv) + 1,
            "n": n,
            "k": k,
            "method": "Rating scale model, shared thresholds (Andrich 1978)",
        }
    )


def cheatsheet():
    return "irtras: rating scale model, item locations b_i with thresholds tau shared across items"
