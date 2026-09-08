# morie.fn -- function file (rootcoder007/morie)
"""Identified bounds on beta when X is discrete (single-index model).

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Section 2.3.2, equation (2.13) (pages 15-16).

When every component of X is discrete, beta is not point identified
(Theorem 2.1 condition (b) fails).  If G is known to be strictly
increasing, though, the support points can be sorted so that

    G(x_1'b) <= G(x_2'b) <= ... <= G(x_M'b)

and tight identified bounds on b_m follow from the linear programs

    maximize (minimize):  b_m
    subject to:           x_j'b <= x_{j+1}'b,  j = 1, ..., M-1   (2.13)

with equality imposed where G(x_j'b) = G(x_{j+1}'b).  Scale
normalisation is b_1 = 1.

Solved with the package's own two-phase simplex (Bland's rule), which
is deterministic: no random pivoting, no restarts.  Free variables are
shifted by an explicit finite `blim` so the standard-form solver sees
only nonnegative variables.
"""

from __future__ import annotations

from . import _array_core as np
from . import _sci_core as sci

from ._richresult import RichResult

__all__ = ["simidentd", "horowitz_sim_id_discrete_x"]


def simidentd(xs, gvals, blim=100.0, tie=1e-12):
    """Identified bounds on the index coefficients under discrete X.

    Parameters
    ----------
    xs : array-like, (M, d)
        The M points of support of X.
    gvals : array-like, (M,)
        E(Y | X = x_m) at each support point.
    blim : float, default 100.0
        Explicit finite box |b_j| <= blim.  A solution at the box edge
        is reported as unbounded -- the book calls such bounds
        uninformative.
    tie : float, default 1e-12
        Two gvals closer than this are treated as equal, which turns
        the corresponding inequality in (2.13) into an equality.

    Returns
    -------
    RichResult
        payload keys: lower, upper, width, bounded, dim, M, method.
    """
    Xs = np.atleast_2d(np.asarray(xs, dtype=float))
    g = np.asarray(gvals, dtype=float).ravel()
    M, d = Xs.shape
    if g.size != M or M < 2 or d < 2:
        return RichResult(payload={
            "lower": np.full(max(d - 1, 0), np.nan),
            "upper": np.full(max(d - 1, 0), np.nan),
            "width": np.full(max(d - 1, 0), np.nan),
            "bounded": [False] * max(d - 1, 0), "dim": d, "M": M,
            "method": "identified bounds (2.13) -- input too small"})

    order = np.argsort(g, kind="stable")
    Xs = Xs[order]
    g = g[order]

    # b = (1, btilde); shift z = btilde + blim >= 0.
    k = d - 1
    rows_ub, rhs_ub, rows_eq, rhs_eq = [], [], [], []
    for j in range(M - 1):
        a = Xs[j, 1:] - Xs[j + 1, 1:]
        c0 = float(Xs[j, 0] - Xs[j + 1, 0])
        # a'btilde + c0 <= 0  ->  a'z <= blim * sum(a) - c0
        row = [float(v) for v in a]
        rhs = blim * float(np.sum(a)) - c0
        if abs(float(g[j + 1] - g[j])) <= tie:
            rows_eq.append(row)
            rhs_eq.append(rhs)
        else:
            rows_ub.append(row)
            rhs_ub.append(rhs)

    A_ub = np.array(rows_ub) if rows_ub else None
    b_ub = np.array(rhs_ub) if rows_ub else None
    A_eq = np.array(rows_eq) if rows_eq else None
    b_eq = np.array(rhs_eq) if rows_eq else None
    box = [(0.0, 2.0 * blim)] * k

    lo = np.full(k, np.nan)
    hi = np.full(k, np.nan)
    bounded = []
    for m in range(k):
        c = np.zeros(k)
        c[m] = 1.0
        r1 = sci.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=box)
        r2 = sci.linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds=box)
        v1 = float(r1["x"][m]) - blim if r1["success"] else np.nan
        v2 = float(r2["x"][m]) - blim if r2["success"] else np.nan
        lo[m] = v1
        hi[m] = v2
        bounded.append(bool(np.isfinite(v1) and np.isfinite(v2)
                            and v1 > -blim + 1e-6 and v2 < blim - 1e-6))
    width = hi - lo
    return RichResult(
        title="Identified bounds on beta, discrete X (eq. 2.13)",
        payload={"lower": lo, "upper": hi, "width": width,
                 "bounded": bounded, "dim": d, "M": M,
                 "method": "Horowitz (2009) eq. (2.13) linear programs"},
    )


horowitz_sim_id_discrete_x = simidentd


def cheatsheet():
    return "hrzsicd: identified bounds on beta when X is discrete (eq. 2.13)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    # Horowitz (2009) Example 2.5 / Table 2.2, page 16: the reported
    # tight bounds are 1 < beta_2 < 1.2.
    xs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.6, 0.5], [1.0, 1.0]]
    g = [0.0, 0.1, 0.3, 0.35, 0.4]
    r = simidentd(xs, g)
    assert abs(float(r["lower"][0]) - 1.0) < 1e-6, r["lower"]
    assert abs(float(r["upper"][0]) - 1.2) < 1e-6, r["upper"]
    print("ok", r["lower"], r["upper"])
