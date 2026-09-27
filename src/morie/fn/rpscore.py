"""Ranked probability score for ordered categorical forecasts (Epstein 1969).

Epstein, E. S. (1969). A scoring system for probability forecasts of ranked categories.
Journal of Applied Meteorology 8, 985-987. Murphy, A. H. (1971). A note on the ranked
probability score. Journal of Applied Meteorology 10, 155-156.
"""

from ._richresult import RichResult

__all__ = ["ranked_probability_score"]


def ranked_probability_score(probs, outcomes):
    r"""RPS = 1/(K - 1) sum_{k=1}^{K-1} (F_k - O_k)^2 per forecast, with F and O the cumulative
    forecast probabilities and the cumulative indicator of the observed category (Epstein 1969,
    in Murphy's 1971 normalised form); 0 is a perfect forecast, 1 the worst.

    Parameters
    ----------
    probs : list of probability vectors over K ordered categories
    outcomes : sequence of observed category indices (0-based)

    Returns
    -------
    RichResult
        Keys: rps (per forecast), mean.

    References
    ----------
    Epstein, E. S. (1969). Journal of Applied Meteorology 8, 985-987.
    Murphy, A. H. (1971). Journal of Applied Meteorology 10, 155-156.

    Examples
    --------
    >>> ranked_probability_score([[0.2, 0.5, 0.3]], [1])["rps"]
    [0.065]
    """
    out = []
    for p, o in zip(probs, outcomes):
        K = len(p)
        F, s, acc = 0.0, 0.0, 0.0
        for k in range(K - 1):
            F += float(p[k])
            O = 1.0 if int(o) <= k else 0.0
            acc += (F - O) ** 2
        out.append(round(acc / (K - 1), 15))
    m = 0.0
    for v in out:
        m += v
    return RichResult(
        title="Ranked probability score",
        summary_lines=[("mean RPS", m / len(out))],
        payload={"rps": out, "mean": m / len(out)},
    )


def cheatsheet():
    return "rpscore: ranked probability score for ordered categorical forecasts"
