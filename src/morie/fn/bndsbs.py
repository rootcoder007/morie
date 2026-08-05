# morie.fn -- function file (rootcoder007/morie)
"""Projection of an identification set onto a subset of coordinates."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["bound_subset_inference"]


def bound_subset_inference(theta_full, subset_idx):
    """Coordinate projection of a d-dimensional identified set.

    Subset inference is a projection: what a joint identified set says
    about one coordinate is the shadow the set casts on that axis.  The
    projection is always at least as wide as any conditional slice, which
    is exactly why projection-based inference is conservative -- the
    reported per-coordinate widths make that visible.

    Formula: ``proj_k(H) = [min_{h in H} h_k, max_{h in H} h_k]`` for each
    selected coordinate ``k``.

    Parameters
    ----------
    theta_full : array-like, shape (m, d)
        Points of the identified set, one per row.
    subset_idx : array-like of int
        Zero-based coordinate indices to project onto.

    Returns
    -------
    RichResult
        ``lower``, ``upper``, ``width`` (first selected coordinate),
        ``total_width`` (sum over selected coordinates), ``max_width``,
        ``d_subset``, ``m``, ``d``.

    References
    ----------
    Romano, J. P. & Shaikh, A. M. (2008).  Inference for identifiable
    parameters in partially identified econometric models.  Journal of
    Statistical Planning and Inference 138(9), 2786-2807.
    doi:10.1016/j.jspi.2008.03.015.  The coverage-of-a-component
    distinction is Section 4.3.3 of Molinari, F. (2021), Handbook of
    Econometrics 7A (arXiv:2004.11751 p. 101).
    """
    M = C.mat(theta_full)
    m = len(M)
    if m == 0:
        raise ValueError("bound_subset_inference: theta_full is empty")
    d = len(M[0])
    for r in M:
        if len(r) != d:
            raise ValueError("bound_subset_inference: ragged theta_full")
    idx = [int(v) for v in C.vec(subset_idx)]
    if not idx:
        raise ValueError("bound_subset_inference: subset_idx is empty")
    for k in idx:
        if k < 0 or k >= d:
            raise ValueError("bound_subset_inference: coordinate index out of range")
    tot = 0.0
    mx = 0.0
    lo0 = hi0 = 0.0
    for pos, k in enumerate(idx):
        col = [r[k] for r in M]
        lo = hi = col[0]
        for v in col:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        w = hi - lo
        tot += w
        if w > mx:
            mx = w
        if pos == 0:
            lo0 = lo
            hi0 = hi
    return RichResult(payload={
        "lower": lo0, "upper": hi0, "width": hi0 - lo0,
        "total_width": tot, "max_width": mx, "d_subset": len(idx),
        "m": m, "d": d,
        "method": "Subset-inference bound"})


def cheatsheet():
    return "bndsbs: Subset-inference bound"
