# morie.fn -- slice s03 (rootcoder007/morie)
"""Functional ANOVA (Sobol) decomposition.

Source consulted: Sobol, I. M. (1993).  Sensitivity estimates for
nonlinear mathematical models.  *Mathematical Modelling and
Computational Experiments* 1(4), 407-414.  On the unit cube with
independent inputs, f admits the unique decomposition

    f(x) = f_0 + sum_i f_i(x_i) + sum_(i<j) f_ij(x_i, x_j) + ...

in which every summand integrates to zero over each of its own
variables; the components follow from

    f_0    = E[f]
    f_i    = E[f | x_i] - f_0
    f_ij   = E[f | x_i, x_j] - f_i - f_j - f_0

and the variances D_S = Var(f_S) sum to the total, sum_S D_S = D.  The
1993 paper was not retrievable here; the decomposition and its
orthogonality are quoted in their standard published form.

The conditional expectations are evaluated on a tensor grid rather than
sampled, so the decomposition is exact to the quadrature error and both
arms agree; the sum-to-total identity is returned as ``closure`` so the
result can be checked rather than trusted.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["fanova_decomposition"]


def fanova_decomposition(f, input_dist=None, d=2, grid=8):
    """First- and second-order fANOVA components on a tensor grid.

    Returns
    -------
    estimate : f_0, the grand mean
    D        : total variance
    D_main   : variance of each first-order component
    D_int    : variance of each pairwise component, in index order (i<j)
    closure  : (sum of component variances) / D, which is 1 for d = 2
    """
    dd = int(d)
    g = int(grid)
    pts = [(t + 0.5) / g for t in range(g)]

    def tf(row):
        if input_dist is None:
            return list(row)
        return [input_dist[a](row[a]) for a in range(dd)]

    idx = [0] * dd
    vals = []
    rows = []
    total = g ** dd
    for c in range(total):
        rem = c
        for a in range(dd - 1, -1, -1):
            idx[a] = rem % g
            rem //= g
        row = [pts[idx[a]] for a in range(dd)]
        rows.append(list(idx))
        vals.append(float(f(tf(row))))
    f0 = k.mean(vals)
    D = 0.0
    for v in vals:
        D += (v - f0) ** 2 / total
    main = []
    mainf = []
    for a in range(dd):
        m = [0.0] * g
        cnt = [0.0] * g
        for c in range(total):
            m[rows[c][a]] += vals[c]
            cnt[rows[c][a]] += 1.0
        comp = [m[t] / cnt[t] - f0 for t in range(g)]
        mainf.append(comp)
        s = 0.0
        for t in range(g):
            s += comp[t] ** 2 / g
        main.append(s)
    inter = []
    for a in range(dd):
        for b in range(a + 1, dd):
            m = [[0.0] * g for _ in range(g)]
            cnt = [[0.0] * g for _ in range(g)]
            for c in range(total):
                m[rows[c][a]][rows[c][b]] += vals[c]
                cnt[rows[c][a]][rows[c][b]] += 1.0
            s = 0.0
            for t in range(g):
                for u in range(g):
                    comp = (m[t][u] / cnt[t][u] - mainf[a][t] - mainf[b][u] - f0)
                    s += comp ** 2 / (g * g)
            inter.append(s)
    acc = 0.0
    for v in main:
        acc += v
    for v in inter:
        acc += v
    return RichResult(
        title="Functional ANOVA decomposition",
        summary_lines=[("f_0", f0), ("D", D)],
        payload={
            "estimate": f0,
            "f0": f0,
            "D": D,
            "D_main": main,
            "D_int": inter,
            "closure": acc / D if D > 0.0 else float("nan"),
            "method": "Sobol (1993) functional ANOVA decomposition on a tensor grid",
        },
    )


def cheatsheet():
    return "fanmd: Sobol decomposition of f"
