# morie.fn -- function file (rootcoder007/morie)
"""Cohen's kappa over spatial neighbour pairs."""

from math import fsum, sqrt

from ._richresult import RichResult
from ._spx import sqmat, vec

__all__ = [
    "spatial_concordance_kappa",
    "spkappa",
]


def spatial_concordance_kappa(x, y, w):
    """Agreement between two categorical maps, scored over neighbour pairs.

    NOT IN SCHABENBERGER & GOTWAY. Fixed-string searches of the book for
    "kappa" and "concordance" return nothing. The agreement coefficient is
    Cohen, J. (1960), "A coefficient of agreement for nominal scales",
    *Educational and Psychological Measurement* 20:37-46, in the
    unweighted form

        kappa = (p_o - p_e) / (1 - p_e).

    What IS taken from the book is the pairing structure. Ordinary kappa
    compares x_i with y_i at the SAME site. Here the comparison is over
    the neighbour pairs of a weights matrix,

        p_o = sum_i sum_j w_ij I{x_i = y_j} / w..,

    which is exactly Mantel's M2, eq (1.5) of Sec. 1.3.1, with
    U_ij = I{x_i = y_j}; Sec. 1.3.2 makes the same substitution to get the
    Black-White join count. So the statistic answers "does a category at a
    site agree with the categories AROUND it", not "do two raters agree".

    p_e is built from the weight-marginal category shares,

        p_e = sum_c px_c qy_c,
        px_c = sum_i (sum_j w_ij) I{x_i = c} / w..,
        qy_c = sum_j (sum_i w_ij) I{y_j = c} / w..,

    i.e. each map's share of the total weight, so that p_e is the
    agreement expected if the two maps were independently shuffled subject
    to the same weight totals.

    kappa = 1 is perfect neighbour agreement, 0 is chance, negative is
    systematic disagreement. kappa is undefined when p_e = 1, which
    happens when both maps are constant, and that raises.

    Parameters
    ----------
    x, y : (n,) array-like
        Category codes at each site (compared as integers).
    w : (n, n) array-like
        Neighbour weights, zero diagonal.

    Returns
    -------
    RichResult
        ``kappa``, ``p_observed``, ``p_expected``, ``categories``,
        ``s0``, ``n``, ``method``.
    """
    xv = vec(x, "x")
    yv = vec(y, "y")
    n = len(xv)
    if len(yv) != n:
        raise ValueError("`x` and `y` must have the same length")
    if n < 2:
        raise ValueError("at least 2 sites are needed")
    xi = [int(round(t)) for t in xv]
    yi = [int(round(t)) for t in yv]
    for t, s in zip(xv, xi):
        if abs(t - s) > 1e-9:
            raise ValueError("`x` must hold integer category codes")
    for t, s in zip(yv, yi):
        if abs(t - s) > 1e-9:
            raise ValueError("`y` must hold integer category codes")
    ww = sqmat(w, n, "w")
    for i in range(n):
        if ww[i][i] != 0.0:
            raise ValueError("`w` must have a zero diagonal")
        for j in range(n):
            if ww[i][j] < 0:
                raise ValueError("`w` must be non-negative for kappa to be "
                                 "a proportion")
    s0 = fsum([fsum(row) for row in ww])
    if s0 <= 0:
        raise ValueError("total weight w.. must be positive")

    cats = sorted(set(xi) | set(yi))
    po = fsum([ww[i][j] for i in range(n) for j in range(n)
               if xi[i] == yi[j]]) / s0
    rows = [fsum(ww[i]) for i in range(n)]
    cols = [fsum([ww[i][j] for i in range(n)]) for j in range(n)]
    pe = fsum([
        (fsum([rows[i] for i in range(n) if xi[i] == c]) / s0)
        * (fsum([cols[j] for j in range(n) if yi[j] == c]) / s0)
        for c in cats])
    if abs(1.0 - pe) < 1e-12:
        raise ValueError("expected agreement is 1; kappa is undefined "
                         "(both maps are effectively constant)")

    return RichResult(payload={
        "kappa": (po - pe) / (1.0 - pe),
        "p_observed": po,
        "p_expected": pe,
        "categories": [float(c) for c in cats],
        "s0": s0,
        "compares_neighbours_not_same_site": True,
        "n": n,
        "method": ("Cohen's kappa (Cohen 1960) over the neighbour pairs of "
                   "Mantel's M2, Schabenberger & Gotway (2005) eq (1.5); "
                   "the kappa coefficient is NOT in that book"),
    })


def cheatsheet():
    return "spcgme: Cohen's kappa over spatial neighbour pairs"


# compact alias per ledger/NAMING.md
spkappa = spatial_concordance_kappa
