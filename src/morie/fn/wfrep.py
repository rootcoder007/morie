# morie.fn -- function file (rootcoder007/morie)
"""Weighted frequency distribution over a categorical variable."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["weighted_frequency"]


def weighted_frequency(y, weights=None, cells=None):
    """Estimated population frequencies of the categories of ``y``.

    Each sampled unit contributes its design weight to the cell it falls
    in, so the cell total estimates the population count rather than the
    sample count.  With every weight equal to one the cell totals are
    exactly the ordinary contingency counts, which is the anchor.

    Formula: ``f_k = sum_{i in cell k} w_i``,
    ``p_k = f_k / sum_k f_k``.

    Parameters
    ----------
    y : array-like
        Category labels (compared as strings, so numeric and character
        labels behave the same in both language arms).
    weights : array-like, optional
        Design weights; equal weights if omitted.
    cells : sequence, optional
        Category levels, in the order to report.  Defaults to the sorted
        distinct labels of ``y``.

    Returns
    -------
    RichResult
        ``estimate`` (largest cell total), ``levels``, ``freq``, ``prop``,
        ``sumw``, ``n``, ``k``, ``method``.

    References
    ----------
    Lohr, S. L. (2010).  Sampling: Design and Analysis, 2nd edition.
    Brooks/Cole, section 7.2.
    """
    lab = [_lab(v) for v in list(y)]
    n = len(lab)
    if n == 0:
        raise ValueError("weighted_frequency: y is empty")
    if weights is None:
        w = [1.0] * n
    else:
        w = C.vec(weights)
    if len(w) != n:
        raise ValueError("weighted_frequency: y and weights differ in length")
    lv = [_lab(v) for v in cells] if cells is not None else sorted(set(lab))
    freq = [0.0] * len(lv)
    idx = {}
    for j, s in enumerate(lv):
        idx.setdefault(s, j)
    for i in range(n):
        j = idx.get(lab[i])
        if j is not None:
            freq[j] += w[i]
    tot = sum(freq)
    prop = [(v / tot if tot > 0.0 else float("nan")) for v in freq]
    return RichResult(payload={
        "estimate": float(max(freq)) if freq else float("nan"),
        "levels": lv, "freq": freq, "prop": prop,
        "sumw": float(sum(w)), "n": n, "k": len(lv),
        "method": "weighted cell frequencies, f_k = sum_{i in k} w_i [Lohr 2010]"})


def _lab(v):
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    return str(v)


# CANONICAL TEST
# >>> r = weighted_frequency([1, 1, 2, 3, 3, 3], None, None)
# >>> assert r["freq"] == [2.0, 1.0, 3.0]      # == as.vector(table(y))
# >>> assert abs(sum(r["prop"]) - 1.0) < 1e-12


def cheatsheet():
    return "wfrep(y, weights, cells): weighted frequency table, f_k = sum w_i."
