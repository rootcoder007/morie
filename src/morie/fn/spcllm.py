# morie.fn -- function file (rootcoder007/morie)
"""LISA cluster classification (HH / LL / HL / LH / NS)."""

from math import fsum, sqrt

from ._richresult import RichResult
from ._spx import mean, sqmat, twosidep, vec

__all__ = [
    "spatial_cluster_lisa",
    "lisaclust",
]


def spatial_cluster_lisa(x, w, alpha=0.05):
    """Classify each site from its local Moran statistic and its sign pair.

    The statistic is eq (1.17) of Schabenberger & Gotway (2005) Sec. 1.3.3,

        I(s_i) = n / {(n-1) S^2} (Z_i - Zbar) sum_j w_ij (Z_j - Zbar),

    and the mean under randomization is the one the book prints,

        E_r[I(s_i)] = -(n-1)^-1 sum_j w_ij.

    THE FOUR-QUADRANT LABELLING IS NOT IN THE BOOK. Fixed-string searches
    for "quadrant" and "Moran scatter" find only an unrelated kriging
    search neighbourhood. HH / LL / HL / LH comes from the Moran
    scatterplot of Anselin, L. (1996), "The Moran scatterplot as an ESDA
    tool", in *Spatial Analytical Perspectives on GIS*, Taylor & Francis,
    pp. 111-125, and the LISA significance filter from Anselin, L. (1995),
    "Local indicators of spatial association -- LISA", *Geographical
    Analysis* 27:93-115, which the book does cite (Sec. 1.3.3, p. 24).
    Only the label scheme is external; the statistic is the book's.

    Inference. The book says Anselin recommends CONDITIONAL randomization
    -- the value at site i is held fixed and only the others are permuted.
    That is done exactly here rather than by sampling, because the
    conditional distribution of the spatial lag has closed-form moments.
    With site i fixed, the remaining n-1 deviations b_1..b_{n-1} are drawn
    without replacement into the neighbour slots, so for
    L_i = sum_j w_ij b_pi(j),

        E[L_i]   = mbar sum_j w_ij
        Var[L_i] = v {sum_j w_ij^2 - (sum_j w_ij)^2 / (n-1)} (n-1)/(n-2)

    with mbar and v the mean and (n-1)-divisor variance of the other n-1
    deviations. This is the standard simple-random-sampling-without-
    replacement result for a weighted sum, DERIVED here, not quoted from
    the book, which prints only the mean of I(s_i).

    ``I(s_i)`` is then standardised through L_i (the leading factor and
    (Z_i - Zbar) are both constants under conditional randomization) and a
    two-sided normal p-value taken. Sites with p >= alpha are "NS".

    Labels for the significant sites use the sign pair (Z_i - Zbar,
    mean neighbour deviation): HH, LL, HL (a high value among low
    neighbours -- a spatial outlier, not a cluster), LH.

    A site with no neighbours gets Var[L_i] = 0; it is labelled "NS" with
    p = 1 rather than dividing by zero.

    Parameters
    ----------
    x : (n,) array-like
        Attribute values.
    w : (n, n) array-like
        Weights with a zero diagonal.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    RichResult
        ``labels``, ``local``, ``z``, ``p_value``, ``counts``,
        ``lagged_mean``, ``n``, ``method``.
    """
    z = vec(x, "x")
    n = len(z)
    if n < 4:
        raise ValueError("at least 4 sites are needed; the conditional "
                         "variance divides by n-2")
    ww = sqmat(w, n, "w")
    for i in range(n):
        if ww[i][i] != 0.0:
            raise ValueError("`w` must have a zero diagonal")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("`alpha` must lie strictly between 0 and 1")
    m = mean(z)
    d = [t - m for t in z]
    ss = fsum([t * t for t in d])
    if ss <= 0:
        raise ValueError("`x` is constant; local Moran's I is undefined")

    labels = []
    local = []
    zs = []
    ps = []
    lagm = []
    for i in range(n):
        li = fsum([ww[i][j] * d[j] for j in range(n)])
        local.append(n * d[i] * li / ss)
        others = [d[j] for j in range(n) if j != i]
        mb = fsum(others) / (n - 1.0)
        v = fsum([(t - mb) ** 2 for t in others]) / (n - 1.0)
        s1 = fsum(ww[i])
        s2 = fsum([ww[i][j] * ww[i][j] for j in range(n)])
        varl = v * (s2 - s1 * s1 / (n - 1.0)) * (n - 1.0) / (n - 2.0)
        nb = fsum([1.0 for j in range(n) if ww[i][j] != 0.0])
        lagm.append(li / nb if nb > 0 else 0.0)
        if varl <= 0:
            zs.append(float("nan"))
            ps.append(1.0)
            labels.append("NS")
            continue
        zi = (li - mb * s1) / sqrt(varl)
        pv = twosidep(zi)
        zs.append(zi)
        ps.append(pv)
        if pv >= alpha:
            labels.append("NS")
        elif d[i] >= 0 and lagm[i] >= 0:
            labels.append("HH")
        elif d[i] < 0 and lagm[i] < 0:
            labels.append("LL")
        elif d[i] >= 0:
            labels.append("HL")
        else:
            labels.append("LH")

    counts = {}
    for k in ("HH", "LL", "HL", "LH", "NS"):
        counts[k] = float(len([t for t in labels if t == k]))

    return RichResult(payload={
        "labels": labels,
        "local": local,
        "z": zs,
        "p_value": ps,
        "lagged_mean": lagm,
        "counts": counts,
        "alpha": alpha,
        "conditional_randomization": True,
        "hl_and_lh_are_outliers_not_clusters": True,
        "n": n,
        "method": ("Local Moran eq (1.17) of Schabenberger & Gotway (2005) "
                   "Sec. 1.3.3 with exact conditional-randomization "
                   "moments; the HH/LL/HL/LH labels are Anselin (1996), "
                   "not in that book"),
    })


def cheatsheet():
    return "spcllm: LISA cluster labels from eq (1.17)"


# compact alias per ledger/NAMING.md
lisaclust = spatial_cluster_lisa
