# morie.fn -- function file (rootcoder007/morie)
r"""Free-Wilson analysis: additive substituent contributions.

**The model.** For a common scaffold with :math:`k` substitution
positions, biological activity is taken to be the sum of the parent
activity and a contribution from whichever group occupies each
position:

.. math:: y_i = \mu + \sum_{p=1}^{k} \sum_{g \in G_p}
          a_{pg} \, I(i \text{ has } g \text{ at } p) + \epsilon_i.

Fit by least squares on the indicator design. Every coefficient is
then a directly readable "this group at this position is worth
:math:`a_{pg}` log units", and activity of an unmade compound is
predicted by adding up its parts.

**The design is singular, always.** Exactly one group occupies each
position, so the indicator columns for a position sum to the
intercept column -- :math:`k` exact linear dependencies before any
data is collected. The fit is therefore *not* unique and the raw
normal equations have no inverse. Two constraints are in use and they
give different numbers for the same fit:

``reference`` drops one group per position, so each remaining
coefficient reads as "relative to the reference group", and
:math:`\mu` is the activity of the all-reference compound. This is
what Free and Wilson's original tables show.

``sum_zero`` requires the contributions at each position to sum to
zero -- weighted by occurrence counts, as in Fujita and Ban's
reformulation -- so :math:`\mu` becomes the mean activity and each
coefficient a deviation from it.

**Fitted values are identical under both.** The constraint fixes
which of the infinitely many solutions is reported; it cannot change
what the model predicts. That is the anchor, and it is the check that
catches a constraint applied to the wrong column.

**What the model cannot do.** Additivity is the assumption, not a
result: a substituent whose effect depends on what sits at another
position is invisible to it, and a group observed at only one
position in one compound has its coefficient determined by that
single compound. Both are reported -- occurrence counts per
coefficient, and the residual degrees of freedom -- rather than left
to be discovered from a suspiciously good fit.

References
----------
Free, S. M. & Wilson, J. W. (1964) "A mathematical contribution to
structure-activity studies", *Journal of Medicinal Chemistry* 7(4),
395-399, doi:10.1021/jm00334a001. The additive model above, the
indicator design over substituent positions, least-squares fitting,
and the singularity that forces a constraint.
"""

import math

from ._richresult import RichResult

__all__ = ["design_matrix", "free_wilson", "predict_activity",
           "CONSTRAINTS"]

CONSTRAINTS = ("reference", "sum_zero")


def _prep(compounds, activity):
    C = []
    for row in compounds:
        C.append(tuple(str(g) for g in row))
    y = [float(v) for v in activity]
    if len(C) != len(y):
        raise ValueError("frwil: %d compounds but %d activities"
                         % (len(C), len(y)))
    if not C:
        raise ValueError("frwil: no compounds given")
    k = len(C[0])
    if k == 0 or any(len(r) != k for r in C):
        raise ValueError("frwil: every compound must list a group for "
                         "the same number of positions")
    return C, y, k


def design_matrix(compounds, constraint="reference"):
    r"""The indicator design, with the chosen constraint applied.

    Returns the matrix, the column names, and the group inventory.
    """
    if constraint not in CONSTRAINTS:
        raise ValueError("frwil: constraint must be one of %s, got %r"
                         % (", ".join(CONSTRAINTS), constraint))
    C = [tuple(str(g) for g in row) for row in compounds]
    if not C:
        raise ValueError("frwil: no compounds given")
    k = len(C[0])
    groups = []
    for p in range(k):
        seen = []
        for row in C:
            if row[p] not in seen:
                seen.append(row[p])
        if len(seen) < 2:
            raise ValueError("frwil: position %d has only the group "
                             "%r, so its contribution cannot be "
                             "separated from the intercept"
                             % (p + 1, seen[0]))
        groups.append(seen)
    names = ["intercept"]
    cols = []
    for p in range(k):
        keep = groups[p][1:] if constraint == "reference" \
            else groups[p]
        for g in keep:
            names.append("P%d:%s" % (p + 1, g))
            cols.append((p, g))
    M = []
    for row in C:
        r = [1.0]
        for p, g in cols:
            r.append(1.0 if row[p] == g else 0.0)
        M.append(r)
    return {"matrix": M, "names": names, "groups": groups,
            "columns": cols, "constraint": constraint,
            "n_positions": k,
            "reference": [g[0] for g in groups]}


def _lstsq(M, y, ridge=0.0):
    """Normal equations by Gauss-Jordan; ridge only breaks ties."""
    n, p = len(M), len(M[0])
    A = [[sum(M[i][a] * M[i][b] for i in range(n))
          + (ridge if a == b else 0.0) for b in range(p)]
         for a in range(p)]
    b = [sum(M[i][a] * y[i] for i in range(n)) for a in range(p)]
    Ab = [A[i] + [b[i]] for i in range(p)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(Ab[r][c]))
        if abs(Ab[piv][c]) < 1e-10:
            raise ValueError("frwil: the design is rank deficient "
                             "even after the constraint -- some "
                             "group appears in no compound that "
                             "distinguishes it")
        Ab[c], Ab[piv] = Ab[piv], Ab[c]
        for r in range(p):
            if r == c:
                continue
            f = Ab[r][c] / Ab[c][c]
            for kk in range(c, p + 1):
                Ab[r][kk] -= f * Ab[c][kk]
    return [Ab[i][p] / Ab[i][i] for i in range(p)]


def free_wilson(compounds, activity, constraint="reference"):
    r"""Fit the additive model and report the substituent values."""
    C, y, k = _prep(compounds, activity)
    D = design_matrix(C, constraint)
    M, names = D["matrix"], D["names"]
    if constraint == "sum_zero":
        # One row per position: the occurrence-weighted contributions
        # at that position must sum to zero (Fujita-Ban form).
        for p in range(k):
            row = [0.0] * len(names)
            for j, (pp, g) in enumerate(D["columns"], start=1):
                if pp == p:
                    row[j] = float(sum(1 for r in C if r[p] == g))
            M = M + [row]
            y = y + [0.0]
        beta = _lstsq(M, y)
        fitted = [sum(D["matrix"][i][j] * beta[j]
                      for j in range(len(beta)))
                  for i in range(len(C))]
    else:
        beta = _lstsq(M, y)
        fitted = [sum(M[i][j] * beta[j] for j in range(len(beta)))
                  for i in range(len(C))]
    resid = [y[i] - fitted[i] for i in range(len(C))]
    mu = sum(y[:len(C)]) / len(C)
    sst = sum((v - mu) ** 2 for v in y[:len(C)])
    sse = sum(r * r for r in resid)
    p_eff = len(beta) - (k if constraint == "sum_zero" else 0)
    df = len(C) - p_eff
    counts = {}
    for j, (pp, g) in enumerate(D["columns"], start=1):
        counts[names[j]] = sum(1 for r in C if r[pp] == g)
    return RichResult(payload={
        "estimate": dict(zip(names, beta)),
        "coefficients": dict(zip(names, beta)),
        "names": names, "beta": beta,
        "fitted": fitted, "residuals": resid,
        "rss": sse, "tss": sst,
        "r_squared": 1.0 - sse / sst if sst > 0 else float("nan"),
        "sigma": math.sqrt(sse / df) if df > 0 else float("nan"),
        "df_residual": df, "n_parameters": p_eff,
        "occurrences": counts,
        "groups": D["groups"], "reference": D["reference"],
        "constraint": constraint, "n_positions": k,
        "method": "Free & Wilson (1964) additive substituent model, "
                  "%s constraint" % constraint,
    })


def predict_activity(fit, compound):
    r"""Predicted activity of a compound, made or unmade."""
    row = tuple(str(g) for g in compound)
    if len(row) != fit["n_positions"]:
        raise ValueError("frwil: the compound lists %d positions but "
                         "the model has %d"
                         % (len(row), fit["n_positions"]))
    coef = fit["coefficients"]
    total = coef["intercept"]
    for p, g in enumerate(row):
        if g not in fit["groups"][p]:
            raise ValueError("frwil: group %r was never observed at "
                             "position %d, so the model says nothing "
                             "about it" % (g, p + 1))
        key = "P%d:%s" % (p + 1, g)
        total += coef.get(key, 0.0)
    return total
