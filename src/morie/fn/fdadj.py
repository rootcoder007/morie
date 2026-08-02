# morie.fn -- function file (rootcoder007/morie)
"""Front-door adjustment formula (discrete)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["frontdoor_adjustment"]


def frontdoor_adjustment(x, z, y, at=None):
    r"""Front-door adjustment of the X -> Y effect through mediator Z.

    .. math:: P(y|do(x)) = \sum_z P(z|x) \sum_{x'} P(y|x', z)\,P(x')

    (Pearl 2009, Thm. 3.3.4). The inner sum re-weights by the MARGINAL
    of X, which is what lets an unobserved X-Y confounder cancel; that
    re-weighting is the whole difference from conditioning. Validity is
    a graph question -- see :func:`morie.fn.fdcrt.frontdoor_criterion`.

    This replaces a placeholder that averaged its first argument. All
    variables are discrete.

    Parameters
    ----------
    x, z, y : array-like, shape (n,)
        Treatment, mediator, outcome.
    at : sequence, optional
        Treatment values to intervene on; default every level.

    Returns
    -------
    RichResult
        keys: ``distribution`` ({x: {y: prob}}), ``incomplete_cells``,
        ``n``, ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality*, 2nd edn. Cambridge UP. Thm. 3.3.4
    (front-door adjustment).
    """
    xa = np.asarray(x).ravel()
    za = np.asarray(z).ravel()
    ya = np.asarray(y).ravel()
    n = xa.size
    if not (za.size == n and ya.size == n):
        raise ValueError(f"x, z and y must share a length; got {n}, {za.size}, {ya.size}.")
    if n == 0:
        raise ValueError("inputs must not be empty.")
    xs = np.unique(xa)
    zs = np.unique(za)
    ys = np.unique(ya)
    p_x = {xv: float(np.mean(xa == xv)) for xv in xs}
    targets = list(xs) if at is None else list(at)
    for t in targets:
        if not np.any(xa == t):
            raise ValueError(f"at = {t!r} does not occur in x.")

    incomplete = []
    dist = {}
    for t in targets:
        acc = {yv: 0.0 for yv in ys}
        sel_t = xa == t
        for zv in zs:
            p_z_x = float(np.mean(za[sel_t] == zv))
            if p_z_x == 0.0:
                continue
            for yv in ys:
                inner = 0.0
                for xv in xs:
                    sel = (xa == xv) & (za == zv)
                    cnt = int(sel.sum())
                    if cnt == 0:
                        incomplete.append((xv, zv))
                        continue
                    inner += float(np.mean(ya[sel] == yv)) * p_x[xv]
                acc[yv] += p_z_x * inner
        dist[t] = acc
    return RichResult(
        payload={
            "distribution": dist,
            "incomplete_cells": sorted(set(incomplete)),
            "n": int(n),
            "method": "Front-door adjustment (Pearl 2009, Thm. 3.3.4), discrete",
        }
    )


def cheatsheet():
    return "fdadj: front-door adjustment P(y|do(x)) via mediator (Pearl Thm 3.3.4)"
