# morie.fn -- function file (rootcoder007/morie)
"""Thermometer folding diagnostics for unfolding analysis."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["folding_problem"]


def folding_problem(ratings, stimulus_order=None):
    r"""Detect the folding signature in thermometer ratings.

    Under the unfolding model :math:`t_{ij} = \alpha - (x_i - y_j)^2`
    each respondent's ratings, read along the stimulus ordering, rise
    to a single peak (their ideal point) and fall -- ratings "fold" at
    the ideal point, which is why raw thermometer scores cannot be
    averaged as if they were positions. Reports each respondent's
    peak, their count of single-peakedness violations, and the share
    of perfectly single-peaked rows.

    Parameters
    ----------
    ratings : array-like, shape (n, q)
        Thermometer ratings (NaN allowed).
    stimulus_order : array-like, optional
        Left-to-right ordering of the stimuli; default the column
        order.

    Returns
    -------
    RichResult
        keys: ``peak`` (n, argmax position in the given order),
        ``violations`` (n, direction changes beyond the single fold),
        ``single_peaked_share``, ``n``, ``q``, ``method``.

    References
    ----------
    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Ch. 4 (unfolding analysis of
    rating-scale data; the folding problem), p. 107.
    """
    R = np.asarray(ratings, dtype=float)
    if R.ndim != 2:
        raise ValueError("ratings must be 2-D (respondents x stimuli).")
    n, q = R.shape
    order = np.arange(q) if stimulus_order is None else np.asarray(stimulus_order, dtype=int)
    if sorted(order.tolist()) != list(range(q)):
        raise ValueError("stimulus_order must be a permutation of 0..q-1.")
    Ro = R[:, order]

    peak = np.full(n, -1)
    viol = np.zeros(n, dtype=int)
    for i in range(n):
        row = Ro[i]
        m = ~np.isnan(row)
        if m.sum() < 3:
            continue
        vals = row[m]
        peak[i] = int(np.argmax(vals))
        d = np.sign(np.diff(vals))
        d = d[d != 0]
        # single-peaked = at most one +to- change and no -to+ change
        changes = np.diff(d)
        viol[i] = int((changes > 0).sum() + max(0, (changes < 0).sum() - 1))

    scored = peak >= 0
    share = float((viol[scored] == 0).mean()) if scored.any() else float("nan")
    return RichResult(
        payload={
            "peak": peak,
            "violations": viol,
            "single_peaked_share": share,
            "n": int(n),
            "q": int(q),
            "method": "Folding diagnostic: single-peakedness along the stimulus order",
        }
    )


def cheatsheet():
    return "foldp: ratings fold at the ideal point -- count single-peakedness violations"


# compact alias per ledger/NAMING.md
foldingproblem = folding_problem
