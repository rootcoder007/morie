# morie.fn -- slice k04 (rootcoder007/morie)
"""Join-count statistics for binary spatial data (Cliff and Ord 1981).

Source FETCHED (reference implementation): ``spdep::joincount.test``
(Bivand, spdep 1.4-2, ``R/jc.R``), whose comments cite Cliff, A. D. and
Ord, J. K. (1981), *Spatial Processes: Models and Applications*, Pion,
page 20, equations (1.31) and (1.32) for free (nonfree) sampling.  The
package source states, with S0, S1, S2 the usual weight constants,
N the number of units and n_k the count of colour k::

    E[J_kk] = S0 n_k (n_k - 1) / (2 N (N-1))                     (1.31)

    4 Var[J_kk] + 4 E[J_kk]^2 =
          S1 n_k(n_k-1) / (N (N-1))
        + (S2 - 2 S1) n_k(n_k-1)(n_k-2) / (N (N-1)(N-2))
        + (S0^2 + S1 - S2) n_k(n_k-1)(n_k-2)(n_k-3)
          / (N (N-1)(N-2)(N-3))                                  (1.32)

so ``Var = 0.25 * (that sum) - E^2``, and the standard deviate is
``(J_kk - E) / sqrt(Var)``.  The weight constants are

    S0 = sum_ij w_ij
    S1 = 0.5 sum_ij (w_ij + w_ji)^2
    S2 = sum_i ( sum_j w_ij + sum_j w_ji )^2 .

Here J_11 = 0.5 sum_ij w_ij x_i x_j, J_00 the same for the zeros, and
the different-colour count J_10 = 0.5 sum_ij w_ij (x_i - x_j)^2.

The previous body of this module was a one-sample Kolmogorov-Smirnov
test against a fitted normal, pasted by the stub generator.  Deleted.
"""

from __future__ import annotations

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["join_count"]


def _weight_constants(W):
    s0 = float(np.sum(W))
    s1 = 0.5 * float(np.sum((W + W.T) ** 2))
    rs = np.sum(W, axis=1)
    cs = np.sum(W, axis=0)
    s2 = float(np.sum((rs + cs) ** 2))
    return s0, s1, s2


def _moments(nk, N, s0, s1, s2):
    if nk < 4 or N < 4:
        return float("nan"), float("nan")
    d1, d2, d3 = N - 1.0, N - 2.0, N - 3.0
    a2 = nk * (nk - 1.0)
    a3 = a2 * (nk - 2.0)
    a4 = a3 * (nk - 3.0)
    e = s0 * a2 / (2.0 * N * d1)
    v = s1 * a2 / (N * d1)
    v += (s2 - 2.0 * s1) * a3 / (N * d1 * d2)
    v += (s0 * s0 + s1 - s2) * a4 / (N * d1 * d2 * d3)
    v = 0.25 * v - e * e
    return float(e), float(v)


def join_count(x, W):
    """Join counts and their standard deviates for a binary map.

    Parameters
    ----------
    x : array-like, shape (N,)
        Binary colour of each unit, coded 0/1.
    W : array-like, shape (N, N)
        Spatial weights matrix; the diagonal is ignored.

    Returns
    -------
    RichResult
        keys: ``BB``, ``WW``, ``BW`` (the three join counts),
        ``E_BB``, ``E_WW``, ``V_BB``, ``V_WW``, ``z_BB``, ``z_WW``,
        ``p_BB``, ``p_WW`` (upper-tail normal p-values), ``S0``,
        ``S1``, ``S2``, ``n``, ``method``.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = int(x.size)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    if W.shape != (N, N):
        raise ValueError("W must be N x N with N = len(x)")
    W = W - np.diag(np.diag(W))
    u = np.unique(x)
    if not np.all((x == 0.0) | (x == 1.0)):
        raise ValueError(f"x must be coded 0/1; saw {u}")

    s0, s1, s2 = _weight_constants(W)
    bb = 0.5 * float(x @ W @ x)
    z = 1.0 - x
    ww = 0.5 * float(z @ W @ z)
    d = x.reshape(N, 1) - x.reshape(1, N)
    bw = 0.5 * float(np.sum(W * d * d))

    n1 = float(np.sum(x))
    n0 = float(N) - n1
    e_bb, v_bb = _moments(n1, float(N), s0, s1, s2)
    e_ww, v_ww = _moments(n0, float(N), s0, s1, s2)
    zb = (bb - e_bb) / np.sqrt(v_bb) if v_bb > 0.0 else float("nan")
    zw = (ww - e_ww) / np.sqrt(v_ww) if v_ww > 0.0 else float("nan")
    return RichResult(
        payload={
            "BB": bb,
            "WW": ww,
            "BW": bw,
            "E_BB": e_bb,
            "E_WW": e_ww,
            "V_BB": v_bb,
            "V_WW": v_ww,
            "z_BB": float(zb),
            "z_WW": float(zw),
            "p_BB": float(stats.norm.sf(zb)) if zb == zb else float("nan"),
            "p_WW": float(stats.norm.sf(zw)) if zw == zw else float("nan"),
            "S0": s0,
            "S1": s1,
            "S2": s2,
            "n": N,
            "method": "Join-count statistics, nonfree sampling (Cliff and Ord 1981 eqs 1.31-1.32)",
        }
    )


def cheatsheet():
    return "jjmsta: join-count statistics for binary spatial data"


# compact alias per ledger/NAMING.md
joincount = join_count
