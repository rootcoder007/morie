# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""D study: project a G study onto a proposed number of conditions.

Brennan (2001), *Generalizability Theory*, Springer, chapter 3.  A D
study holds the estimated variance components fixed and asks what the
coefficients would be for a different number of conditions n' of the
measurement facet:

    E rho^2(n') = sigma^2_p / (sigma^2_p + sigma^2_pi / n'),
    Phi(n')     = sigma^2_p / (sigma^2_p + (sigma^2_i + sigma^2_pi) / n').

Both increase monotonically in n' and tend to 1, which is what makes
the D study a decision tool: it prices the reliability gain of a
longer instrument.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["d_study_decision"]


def d_study_decision(G_components, n_proposed, target=0.8):
    """Coefficients at each proposed facet size, and the size that meets target.

    Parameters
    ----------
    G_components : (sigma^2_p, sigma^2_i, sigma^2_pi)
    n_proposed : int or array-like of int
        One or more candidate numbers of conditions.
    target : float
        Generalizability coefficient the decision looks for.
    """
    g = core.vec(G_components)
    if len(g) != 3:
        raise ValueError("d_study_decision: G_components must hold three variances")
    vp, vi, vpi = g[0], g[1], g[2]
    for v in g:
        if v < 0:
            raise ValueError("d_study_decision: variance components must be non-negative")
    ns = [int(v) for v in core.vec(n_proposed)]
    if not ns:
        raise ValueError("d_study_decision: n_proposed is empty")
    for v in ns:
        if v < 1:
            raise ValueError("d_study_decision: n_proposed must be positive")
    if not (0.0 < float(target) < 1.0):
        raise ValueError("d_study_decision: target must lie in (0, 1)")
    er = []
    ph = []
    for k in ns:
        de = vp + vpi / k
        er.append(vp / de if de != 0 else float("nan"))
        de2 = vp + (vi + vpi) / k
        ph.append(vp / de2 if de2 != 0 else float("nan"))
    meets = [1 if (v == v and v >= float(target)) else 0 for v in er]
    chosen = 0
    for i in range(len(ns)):
        if meets[i] == 1:
            chosen = ns[i]
            break
    return RichResult(
        title="D study",
        summary_lines=[("candidates", len(ns)), ("target", float(target))],
        payload={
            "estimate": er[-1],
            "e_rho2": er,
            "phi": ph,
            "meets_target": meets,
            "n_required": chosen,
            "n": len(ns),
            "method": "D study projection of fixed variance components, Brennan (2001) ch. 3",
        },
    )


def cheatsheet():
    return "genvdm: D-study decision"


# compact alias per ledger/NAMING.md
dstudydecision = d_study_decision
