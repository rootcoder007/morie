# morie.fn -- function file (rootcoder007/morie)
"""Outlier channel split with per-set bit allocation."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["outsplit", "turboquant_outlier_channel_split"]


def outsplit(x, b_out=8, b_in=2, frac=0.01):
    """Split coordinates into outlier and non-outlier sets, bits allocated apart.

    A handful of large-magnitude channels dominate the quantization
    error of a transformer activation, and a single codebook has to
    stretch to cover them at the cost of resolution everywhere else.
    Separating them buys back that resolution for the bulk at a small
    average cost -- and the average bit-width is what comes out
    NON-INTEGER, which is the whole point of the construction.

    The split is by MAGNITUDE RANK, not by an absolute threshold, so
    the outlier count is exactly ceil(frac . d) regardless of scale and
    the two language arms select the same channels.  Ties break on the
    lower index.

    Formula: O = the ceil(frac d) coordinates of largest |x_j|;
             effective bits = (|O| b_out + (d - |O|) b_in) / d

    Parameters
    ----------
    x : array-like
        The vector whose channels are split.
    b_out : int
        Bits per outlier coordinate.
    b_in : int
        Bits per non-outlier coordinate.
    frac : float
        Fraction of coordinates treated as outliers, in (0, 1).

    Returns
    -------
    RichResult
        ``outlier_index`` (one-based), ``n_outlier``,
        ``effective_bits``, ``outlier_energy`` (share of ||x||^2 in
        the outlier set), ``threshold``, ``d``.

    References
    ----------
    Zandieh et al., TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate, arXiv:2504.19874: "non-integer bit
    precisions result from our strategy of splitting channels into
    outlier and non-outlier sets, and applying two independent
    instances of TurboQuant to each, allocating higher bit precision to
    outliers".  Fetched from arXiv.  The paper does not fix the split
    RULE; magnitude-rank selection is used here and documented as such
    rather than presented as the paper's.
    """
    x = C.vec(x)
    d = len(x)
    if d < 1:
        raise ValueError("the vector must be non-empty")
    bo = int(b_out)
    bi = int(b_in)
    if bo < 1 or bi < 1:
        raise ValueError("both bit widths must be at least 1")
    if bo < bi:
        raise ValueError("outliers must not get fewer bits than the bulk")
    f = float(frac)
    if not 0.0 < f < 1.0:
        raise ValueError("frac must lie strictly between 0 and 1")
    k = int(math.ceil(f * d))
    if k < 1:
        k = 1
    if k >= d:
        raise ValueError("frac selects every coordinate as an outlier")
    order = sorted(range(d), key=lambda i: (-abs(x[i]), i))
    sel = sorted(order[:k])
    tot = sum(v * v for v in x)
    eo = sum(x[i] ** 2 for i in sel)
    return RichResult(payload={
        "outlier_index": [i + 1 for i in sel], "n_outlier": float(k),
        "effective_bits": (k * bo + (d - k) * bi) / d,
        "outlier_energy": eo / tot if tot > 0 else float("nan"),
        "threshold": abs(x[order[k - 1]]), "d": float(d),
        "method": "Outlier channel split with per-set bit allocation"})


turboquant_outlier_channel_split = outsplit


def cheatsheet():
    return "tqoutl: top-ceil(frac d) channels by |x| get b_out bits, rest b_in"
