# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P3: a hot-spot boundary and the spectral gap (``research/lean/P3Cheeger.lean``; Chung, Theorem 1).

* ``Research.P3.testVec_orth`` / ``testVec_dnorm`` / ``testVec_dirichlet`` / ``rayleigh_testVec``
* ``Research.P3.rayleigh_le_two_conductance`` / ``lambda2_le_rayleigh_testVec`` / ``cheeger_easy``

R parity: ``rmorie`` ``R/cheeger.R`` (``morie_cheeger_bound``).
"""

from __future__ import annotations

from morie.fn import _array_core as np

__all__ = ["cheeger_bound"]


def cheeger_bound(adjacency, S, exhaustive=False) -> dict:
    """Conductance of a set and the Cheeger bound ``lambda_2 <= E(f)/D(f) <= 2 h(S)``.

    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
    ...      [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]]
    >>> r = cheeger_bound(A, S=[0, 1, 2], exhaustive=True)
    >>> (round(r["conductance"], 12), round(r["rayleigh_test"], 12), r["bound_holds"], r["argmin_set"])
    (0.142857142857, 0.285714285714, True, [0, 1, 2])
    """
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("adjacency must be square")
    if np.any(A < 0) or float(np.max(np.abs(A - A.T))) > 1e-12:
        raise ValueError("adjacency must be symmetric and non-negative")
    S_list = list(S)
    if (
        len(S_list) == n
        and all(isinstance(v, bool) or v in (True, False) for v in S_list)
        and any(isinstance(v, bool) for v in S_list)
    ):
        in_s = [bool(v) for v in S_list]
    else:
        idx = set(int(v) for v in S_list)
        in_s = [i in idx for i in range(n)]
    if not any(in_s) or all(in_s):
        raise ValueError("S must be a non-empty proper subset of the places")
    d = A.sum(axis=1)
    if np.any(d <= 0):
        raise ValueError("every place needs positive degree")

    def stats_of(mask):
        vol_s = float(sum(d[i] for i in range(n) if mask[i]))
        vol_c = float(sum(d[i] for i in range(n) if not mask[i]))
        cut_s = float(sum(A[i, j] for i in range(n) if mask[i] for j in range(n) if not mask[j]))
        return cut_s, vol_s, vol_c, cut_s / min(vol_s, vol_c), cut_s * (1 / vol_s + 1 / vol_c)

    cut_s, vol_s, vol_c, h, ray = stats_of(in_s)
    dis = 1 / np.sqrt(d)
    L = np.eye(n) - np.outer(dis, dis) * A
    lam = sorted(float(v) for v in np.linalg.eigvalsh((L + L.T) / 2))
    lambda2 = lam[1]
    out = {
        "cut": cut_s,
        "vol_S": vol_s,
        "vol_complement": vol_c,
        "conductance": h,
        "rayleigh_test": ray,
        "lambda2": lambda2,
        "bound_holds": bool(lambda2 <= ray + 1e-10 and ray <= 2 * h + 1e-10),
    }
    if exhaustive:
        if n > 16:
            raise ValueError("exhaustive search is limited to 16 places")
        best = float("inf")
        arg = None
        for code in range(1, 2 ** (n - 1)):
            mask = [bool(code & (1 << i)) for i in range(n)]
            hh = stats_of(mask)[3]
            if hh < best:
                best = hh
                arg = [i for i in range(n) if mask[i]]
        out["cheeger_constant"] = best
        out["argmin_set"] = arg
        out["cheeger_bound_holds"] = bool(lambda2 <= 2 * best + 1e-10)
    out["theorems"] = [
        "Research.P3.testVec_orth",
        "Research.P3.testVec_dnorm",
        "Research.P3.testVec_dirichlet",
        "Research.P3.rayleigh_testVec",
        "Research.P3.rayleigh_le_two_conductance",
        "Research.P3.lambda2_le_rayleigh_testVec",
        "Research.P3.cheeger_easy",
    ]
    return out
