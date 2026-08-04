# morie.fn -- slice s03 (rootcoder007/morie)
"""Warm-started Sinkhorn from previously optimised potentials.

Source consulted (FETCHED): Schmitzer, B. (2019).  Stabilized sparse
scaling algorithms for entropy regularized transport problems.  *SIAM
Journal on Scientific Computing* 41(3), A1443-A1481 (arXiv:1610.06519),
whose epsilon-scaling scheme *is* warm starting: the potentials solved
at one regularisation are used as the initialisation at the next, which
is what makes small-epsilon problems tractable at all.  The scaling
factors are recovered by u = exp(f / eps), v = exp(g / eps), which is
the substitution the paper's section 3 makes explicit.

The saving is reported rather than asserted: ``n_iter`` and
``n_iter_cold`` are the iteration counts with and without the warm
start, so the benefit of the supplied potentials is visible.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .otsklog import ot_sinkhorn_log

__all__ = ["ot_optimised_potentials_warm"]


def ot_optimised_potentials_warm(a, b, C, epsilon=0.1, f0=None, g0=None,
                                 max_iter=200, tol=1e-13):
    """Resume Sinkhorn from (f0, g0), and report what the warm start saved.

    Returns
    -------
    RichResult with payload:
        T, f, g   : plan and potentials
        estimate  : the transport cost
        u, v      : the scaling factors exp(f/eps), exp(g/eps)
        n_iter, n_iter_cold
    """
    warm = ot_sinkhorn_log(a, b, C, epsilon, max_iter, tol, f0, g0)
    cold = ot_sinkhorn_log(a, b, C, epsilon, max_iter, tol, None, None)
    e = float(epsilon)
    u = [math.exp(x / e) for x in warm["f"]]
    v = [math.exp(x / e) for x in warm["g"]]
    return RichResult(
        title="Warm-started Sinkhorn",
        summary_lines=[("iterations", warm["n_iter"]),
                       ("cold-start iterations", cold["n_iter"])],
        payload={
            "T": warm["T"],
            "f": warm["f"],
            "g": warm["g"],
            "estimate": warm["cost"],
            "cost": warm["cost"],
            "u": u,
            "v": v,
            "n_iter": warm["n_iter"],
            "n_iter_cold": cold["n_iter"],
            "saved": cold["n_iter"] - warm["n_iter"],
            "method": "Warm-started log-domain Sinkhorn (Schmitzer 2019 epsilon-scaling)",
        },
    )


def cheatsheet():
    return "otopw: Warm-started Sinkhorn from previous (f,g)"
