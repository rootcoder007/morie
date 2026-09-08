# morie.fn -- function file (rootcoder007/morie)
"""Pair correlation function R(h) = K'(h)/(2 pi h), Sec. 3.4.1."""

from math import fsum, pi

from ._richresult import RichResult
from ._spx import eucdist, mat, vec

__all__ = [
    "schabenberger_pair_correlation",
    "pcf",
]


def schabenberger_pair_correlation(points, region=None, r=None,
                                   correction="border"):
    """Pair correlation function of a mapped point pattern.

    Schabenberger & Gotway (2005) Sec. 3.4.1 defines Ripley's K-function
    at eq (3.7) and then lists the pair-correlation function as one of the
    functions of lambda_2 that relate to it:

        R(h) = (1 / (2 h pi)) dK(h)/dh.

    (The book writes it R(h); the ecological literature writes g(r). Same
    function.) For a homogeneous Poisson process K(h) = pi h^2, so
    R(h) = 1; R(h) > 1 is clustering at scale h and R(h) < 1 regularity.

    K is estimated as in Sec. 3.4.2. The naive moment estimator there is

        Etilde(h) = n^-1 sum_i sum_{j != i} I(h_ij <= h),
        Ktilde(h) = Etilde(h) / lambdahat,   lambdahat = N(A)/nu(A)  eq (3.8)

    and the book states plainly that it is NEGATIVELY BIASED because events
    outside the study region are not observed. The default here is
    therefore the book's border correction, which keeps only events whose
    distance to the boundary exceeds h,

        Ehat*(h) = sum_i sum_{j != i} I(h_ij <= h and d_j > h)
                   / sum_j I(d_j > h);

    pass ``correction="none"`` for the naive estimator instead.

    The derivative is taken by central differences on the supplied radii,
    forward/backward at the ends. A pair correlation function is a
    derivative of a step function estimate, so it is noisier than K itself;
    that is a property of the estimator, not of this implementation.

    Parameters
    ----------
    points : (n, 2) array-like
        Event coordinates.
    region : ((xlo, xhi), (ylo, yhi)), optional
        Rectangular study region; defaults to the bounding box of `points`.
    r : sequence of float, optional
        Distances at which to evaluate; defaults to 10 steps up to a
        quarter of the shorter region side.
    correction : {"border", "none"}
        Edge correction for Khat.

    Returns
    -------
    RichResult
        ``r``, ``k``, ``pcf``, ``lambda``, ``area``, ``csr_k``, ``n``,
        ``method``.
    """
    p = mat(points, "points")
    if len(p[0]) < 2:
        raise ValueError("`points` must have at least two coordinate columns")
    n = len(p)
    if n < 3:
        raise ValueError("at least 3 events are needed")
    if region is None:
        xs = [q[0] for q in p]
        ys = [q[1] for q in p]
        reg = [[min(xs), max(xs)], [min(ys), max(ys)]]
    else:
        reg = mat(region, "region")
        if len(reg) != 2 or len(reg[0]) != 2 or len(reg[1]) != 2:
            raise ValueError("`region` must be ((xlo, xhi), (ylo, yhi))")
    wid = reg[0][1] - reg[0][0]
    hgt = reg[1][1] - reg[1][0]
    if wid <= 0 or hgt <= 0:
        raise ValueError("`region` must have positive width and height")
    area = wid * hgt
    lam = n / area

    if r is None:
        top = 0.25 * min(wid, hgt)
        rr = [top * (k + 1) / 10.0 for k in range(10)]
    else:
        rr = vec(r, "r")
        if any(t <= 0 for t in rr):
            raise ValueError("`r` must be positive")
        if any(rr[k] <= rr[k - 1] for k in range(1, len(rr))):
            raise ValueError("`r` must increase")
    if len(rr) < 2:
        raise ValueError("at least 2 radii are needed to difference K")

    dist = [[eucdist(p[i][:2], p[j][:2]) for j in range(n)] for i in range(n)]
    bdist = [min(p[i][0] - reg[0][0], reg[0][1] - p[i][0],
                 p[i][1] - reg[1][0], reg[1][1] - p[i][1]) for i in range(n)]

    kv = []
    for h in rr:
        if correction == "none":
            cnt = fsum([1.0 for i in range(n) for j in range(n)
                        if i != j and dist[i][j] <= h])
            kv.append((cnt / n) / lam)
        elif correction == "border":
            keep = [j for j in range(n) if bdist[j] > h]
            if not keep:
                kv.append(float("nan"))
                continue
            cnt = fsum([1.0 for i in range(n) for j in keep
                        if i != j and dist[i][j] <= h])
            kv.append((cnt / len(keep)) / lam)
        else:
            raise ValueError('`correction` must be "border" or "none"')

    g = []
    m = len(rr)
    for k in range(m):
        if k == 0:
            der = (kv[1] - kv[0]) / (rr[1] - rr[0])
        elif k == m - 1:
            der = (kv[m - 1] - kv[m - 2]) / (rr[m - 1] - rr[m - 2])
        else:
            der = (kv[k + 1] - kv[k - 1]) / (rr[k + 1] - rr[k - 1])
        g.append(der / (2.0 * pi * rr[k]))

    return RichResult(payload={
        "r": rr,
        "k": kv,
        "pcf": g,
        "lambda": lam,
        "area": area,
        "csr_k": [pi * t * t for t in rr],
        "csr_pcf_is_one": True,
        "correction": correction,
        "n": n,
        "method": ("Pair correlation R(h)=K'(h)/(2 pi h), Schabenberger & "
                   "Gotway (2005) Sec. 3.4.1, with Khat of Sec. 3.4.2 "
                   "and eq (3.8)"),
    })


def cheatsheet():
    return "sppair: pair correlation function R(h)=K'(h)/(2 pi h)"


# compact alias per ledger/NAMING.md
pcf = schabenberger_pair_correlation
