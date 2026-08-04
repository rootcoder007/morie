# morie.fn -- slice s03 (rootcoder007/morie)
"""RetNet's retention mechanism, in both its forms.

Source consulted (FETCHED): Sun, Y. et al. (2023).  Retentive network: a
successor to transformer for large language models.  arXiv:2307.08621.
The recurrent form, the paper's equation (5)-(6), is

    S_n = gamma S_(n-1) + K_n^T V_n
    O_n = Q_n S_n

and the parallel form contracts the same computation into a masked
matrix product,

    Retention(X) = (Q K^T (x) D) V,   D_(nm) = gamma^(n-m) if n >= m else 0

so the two agree exactly.  The paper's whole claim is that they do, so
both are computed here and their maximum discrepancy is returned as
``max_gap``: it is the check that the identity actually holds in this
implementation, not merely in the paper.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["retnet_retention"]


def retnet_retention(y, Q=None, K=None, V=None, gamma=0.9):
    """Retention output, computed recurrently and in parallel.

    Parameters
    ----------
    y : 2-D array-like
        Alternative slot for Q.  (First, for signature stability.)
    Q, K, V : 2-D array-like
        Query, key and value sequences, one row per time step.
    gamma : float
        The decay.

    Returns
    -------
    RichResult with payload:
        estimate  : O[0][0]
        out       : the recurrent output, one row per step
        out_par   : the parallel-form output
        max_gap   : max |recurrent - parallel| over all entries
        state     : the final S_n
    """
    Qm = k.mat(Q if Q is not None else y)
    Km = k.mat(K)
    Vm = k.mat(V)
    n = len(Qm)
    dk = len(Qm[0]) if n else 0
    dv = len(Vm[0]) if Vm else 0
    g = float(gamma)
    S = [[0.0] * dv for _ in range(dk)]
    out = []
    for t in range(n):
        for a in range(dk):
            for b in range(dv):
                S[a][b] = g * S[a][b] + Km[t][a] * Vm[t][b]
        row = [0.0] * dv
        for b in range(dv):
            s = 0.0
            for a in range(dk):
                s += Qm[t][a] * S[a][b]
            row[b] = s
        out.append(row)
    par = []
    for t in range(n):
        row = [0.0] * dv
        for m in range(t + 1):
            qk = 0.0
            for a in range(dk):
                qk += Qm[t][a] * Km[m][a]
            w = qk * (g ** (t - m))
            for b in range(dv):
                row[b] += w * Vm[m][b]
        par.append(row)
    gap = 0.0
    for t in range(n):
        for b in range(dv):
            d = abs(out[t][b] - par[t][b])
            if d > gap:
                gap = d
    return RichResult(
        title="RetNet retention",
        summary_lines=[("steps", n), ("max gap", gap)],
        payload={
            "estimate": out[0][0] if out and out[0] else float("nan"),
            "out": out,
            "out_par": par,
            "max_gap": gap,
            "state": S,
            "gamma": g,
            "method": "RetNet retention, recurrent and parallel forms (Sun et al. 2023, eqs. 5-6)",
        },
    )


def cheatsheet():
    return "retnet: RetNet retention mechanism (gated recurrent + parallel)"
