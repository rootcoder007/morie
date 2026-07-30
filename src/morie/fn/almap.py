# morie.fn -- function file (rootcoder007/morie)
"""Mean average precision."""

import numpy as np

from ._richresult import RichResult

__all__ = ["mean_average_precision", "alammar_mean_average_precision"]


def mean_average_precision(relevance, k=None):
    r"""Mean average precision over a set of queries.

    For one query with ranked binary relevance :math:`r_1, r_2, \ldots`,

    .. math::
       AP = \frac{1}{R}\sum_{i=1}^{n} r_i \cdot P@i,
       \qquad P@i = \frac{1}{i}\sum_{j\le i} r_j,

    and MAP is the mean of :math:`AP` over queries. The normaliser
    :math:`R` is the number of relevant documents FOR THAT QUERY, which
    is what makes queries with different numbers of relevant results
    comparable.

    MAP rewards ranking relevant documents early, and it does so
    smoothly: moving a relevant document from rank 5 to rank 2 raises
    the score, where precision@k would not notice unless the move
    crossed the cutoff. That is the reason to prefer it over
    precision@k for retrieval evaluation.

    Its blind spot is that relevance is BINARY. A document that is
    exactly what the user wanted and one that is marginally on-topic
    count the same, so MAP cannot distinguish a ranking that puts the
    best result first from one that puts a mediocre relevant result
    first. nDCG exists for graded relevance and is the right measure
    when the grades are available.

    Parameters
    ----------
    relevance : sequence of array-like
        Per query, binary relevance in rank order.
    k : int, optional
        Truncate each ranking to the first ``k``.

    Returns
    -------
    RichResult
        ``map``, ``average_precisions``, ``precision_at_k``,
        ``recall_at_k``, ``n_queries``, ``queries_without_relevant``.

    References
    ----------
    Alammar and Grootendorst (2024), *Hands-On Large Language Models*,
    O'Reilly, chapter 8, mean average precision.
    Manning, Raghavan and Schutze (2008), *Introduction to Information
    Retrieval*, section 8.4.

    Examples
    --------
    >>> float(mean_average_precision([[1, 0, 1]])["map"])
    0.8333333333333333
    """
    if len(relevance) == 0:
        raise ValueError("need at least one query.")
    aps, pk, rk = [], [], []
    empty = 0
    for q, rel in enumerate(relevance):
        r = np.asarray(rel, dtype=float).ravel()
        if r.size == 0:
            raise ValueError("query %d has an empty ranking." % q)
        if not np.all(np.isin(r, (0.0, 1.0))):
            raise ValueError(
                "relevance must be binary 0/1; query %d is not." % q
            )
        total_rel = float(r.sum())
        rr = r if k is None else r[: int(k)]
        if total_rel == 0:
            empty += 1
            aps.append(np.nan)
            pk.append(float(rr.mean()))
            rk.append(np.nan)
            continue
        idx = np.arange(1, rr.size + 1)
        prec = np.cumsum(rr) / idx
        aps.append(float(np.sum(rr * prec) / total_rel))
        pk.append(float(rr.mean()))
        rk.append(float(rr.sum() / total_rel))
    aps = np.asarray(aps)
    good = ~np.isnan(aps)
    return RichResult(
        payload={
            "estimate": float(np.mean(aps[good])) if good.any() else np.nan,
            "map": float(np.mean(aps[good])) if good.any() else np.nan,
            "average_precisions": aps,
            "precision_at_k": np.asarray(pk),
            "recall_at_k": np.asarray(rk),
            "k": None if k is None else int(k),
            "n_queries": int(len(relevance)),
            "queries_without_relevant": empty,
            "empty_note": (
                None if empty == 0 else
                "%d quer(y/ies) had no relevant document at all; AP is "
                "undefined there and they are excluded from the mean rather "
                "than scored zero" % empty
            ),
            "normalisation_note": (
                "AP divides by the number of relevant documents for THAT "
                "query, which is what makes queries with different numbers "
                "of relevant results comparable"
            ),
            "binary_note": (
                "relevance is binary here, so a perfect match and a "
                "marginally on-topic document score the same; use nDCG when "
                "graded relevance is available"
            ),
            "method": "Mean average precision",
        }
    )


def cheatsheet():
    return (
        "almap: MAP over queries, with precision/recall at k and the binary "
        "relevance limitation stated"
    )


#: Catalogue alias for :func:`mean_average_precision`.
alammar_mean_average_precision = mean_average_precision
