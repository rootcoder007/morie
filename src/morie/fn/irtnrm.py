"""Bock nominal response model across items with category-specific slopes."""

from __future__ import annotations

from ._irtcore import as_matrix, cat_moments, nrm_probs, seq_
from ._richresult import RichResult

__all__ = ["nominal_response"]


def nominal_response(theta, a, c):
    r"""Nominal response model for a set of items, each with its own slopes.

    For item :math:`i` and category :math:`r`,

    .. math::
        P(X_{vi} = r) = \frac{\exp(a_{ir}\theta_v + c_{ir})}
                             {\sum_s \exp(a_{is}\theta_v + c_{is})} .

    The category-specific slopes :math:`a_{ir}` are what make this a
    nominal rather than an ordinal model: nothing constrains them to be
    ordered.

    The previous body was a placeholder: it averaged a leading ``X``
    argument and used ``ncats`` for nothing. Both are gone.

    Parameters
    ----------
    theta : float or array-like
        Person abilities, length ``n``.
    a : array-like
        ``k`` by ``ncat`` matrix of category slopes, one row per item.
    c : array-like
        ``k`` by ``ncat`` matrix of category intercepts, same shape as ``a``.

    Returns
    -------
    RichResult
        ``p`` (a list of ``k`` matrices, each ``n`` by ``ncat``),
        ``expected`` (``n`` by ``k`` mean slopes), ``info`` (``n`` by ``k``
        item informations, each the variance of :math:`a_{iR}`),
        ``test_info``, ``theta``, ``a``, ``c``, ``ncat``, ``n``, ``k``,
        ``method``.

    Notes
    -----
    The model is invariant to adding a constant to a whole row of ``a`` or
    of ``c``; no identification constraint is imposed here because none is
    needed to evaluate the probabilities.

    References
    ----------
    Bock, R. D. (1972). Estimating item parameters and latent ability when
    responses are scored in two or more nominal categories. *Psychometrika*,
    37(1), 29-51. doi:10.1007/BF02291411

    Tutz, G. (2020). A taxonomy of polytomous item response models.
    arXiv:2010.01382, eq. (14), p. 16.
    """
    th = [float(v) for v in seq_(theta)]
    n = len(th)
    if n == 0:
        raise ValueError("theta is empty.")
    am = as_matrix(a, "a")
    cm = as_matrix(c, "c")
    k = len(am)
    if len(cm) != k or len(cm[0]) != len(am[0]):
        raise ValueError("c must have the same shape as a.")
    ncat = len(am[0])
    if ncat < 2:
        raise ValueError("a needs at least two categories.")

    p = []
    expected = [[0.0] * k for _ in range(n)]
    info = [[0.0] * k for _ in range(n)]
    for i in range(k):
        rows = []
        for v in range(n):
            pr, _ = nrm_probs(th[v], am[i], cm[i])
            mu, var = cat_moments(pr, am[i])
            rows.append(pr)
            expected[v][i] = mu
            info[v][i] = var
        p.append(rows)

    return RichResult(
        payload={
            "p": p,
            "expected": expected,
            "info": info,
            "test_info": [sum(r) for r in info],
            "theta": th,
            "a": am,
            "c": cm,
            "ncat": ncat,
            "n": n,
            "k": k,
            "method": "Nominal response model, category-specific slopes (Bock 1972)",
        }
    )


def cheatsheet():
    return "irtnrm: Bock NRM over items, slopes a_ir and intercepts c_ir per category"
