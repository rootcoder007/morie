# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P15: sentencing disparity decompositions (``research/lean/P15Decomposition.lean``; Oaxaca 1973; Blinder 1973).

* ``Research.P15.twofold_B`` / ``twofold_A`` / ``threefold``
* ``Research.P15.reference_dependence`` / ``explained_eq_iff`` / ``attribution_shift``

R parity: ``rmorie`` ``R/disparity_decomposition.R`` (``morie_disparity_decomposition``).
"""

from __future__ import annotations

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = ["disparity_decomposition", "dfl_reweight"]


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(beta, dtype=float)


def disparity_decomposition(y, X, group, reference, names=None, shift=1.0) -> dict:
    """Oaxaca-Blinder decomposition of a gap in means, with both references and the interaction.

    ``X`` holds the covariates (one column each, no constant: it is added), ``group``
    the two-level label per row, ``reference`` the level whose coefficients price
    the explained part (group B).

    Examples
    --------
    >>> y = [10, 12, 13, 15, 9, 8, 11, 7]; X = [[1, 0], [2, 1], [3, 1], [4, 0], [1, 1], [2, 0], [2, 0], [3, 1]]
    >>> r = disparity_decomposition(y, X, ["A"] * 4 + ["B"] * 4, reference="B", names=["x1", "x2"])
    >>> (round(r["gap"], 12), [round(abs(v), 12) for v in r["identity_checks"].values()])
    (3.75, [0.0, 0.0, 0.0, 0.0])
    >>> ({k: round(v, 12) for k, v in r["twofold_B"].items()}, round(r["threefold"]["interaction"], 12))
    ({'explained': -0.5, 'unexplained': 4.25}, 1.3)
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    g = [str(v) for v in group]
    if y.shape[0] != n or len(g) != n:
        raise ValueError("y, X and group must describe the same rows")
    lv = sorted(set(g))
    if len(lv) != 2:
        raise ValueError("group must take exactly two values")
    B = str(reference)
    if B not in lv:
        raise ValueError("reference must be one of the two group levels")
    A = lv[0] if lv[1] == B else lv[1]
    if names is None:
        names = [f"x{j + 1}" for j in range(k)]
    names = ["(Intercept)", *[str(v) for v in names]]
    D = np.column_stack([np.ones(n), X])
    ia = [i for i in range(n) if g[i] == A]
    ib = [i for i in range(n) if g[i] == B]
    DA = D[ia, :]
    DB = D[ib, :]
    if int(np.linalg.matrix_rank(DA)) < k + 1 or int(np.linalg.matrix_rank(DB)) < k + 1:
        raise ValueError("the covariates are collinear within a group")
    bA = _ols(DA, y[ia])
    bB = _ols(DB, y[ib])
    xA = DA.mean(axis=0)
    xB = DB.mean(axis=0)
    yA = float(np.mean(y[ia]))
    yB = float(np.mean(y[ib]))
    explained_B = float(np.sum((xA - xB) * bB))
    unexplained_A = float(np.sum(xA * (bA - bB)))
    explained_A = float(np.sum((xA - xB) * bA))
    unexplained_B = float(np.sum(xB * (bA - bB)))
    interaction = float(np.sum((xA - xB) * (bA - bB)))
    gap = yA - yB
    return {
        "levels": {"A": A, "B": B},
        "means": {"A": yA, "B": yB},
        "gap": gap,
        "twofold_B": {"explained": explained_B, "unexplained": unexplained_A},
        "twofold_A": {"explained": explained_A, "unexplained": unexplained_B},
        "threefold": {"endowments": explained_B, "coefficients": unexplained_B, "interaction": interaction},
        "by_variable": pd.DataFrame(
            {
                "variable": names,
                "explained_B": (xA - xB) * bB,
                "unexplained_A": xA * (bA - bB),
                "unexplained_shift": float(shift) * (bA - bB),
            }
        ),
        "identity_checks": {
            "twofold_B": gap - (explained_B + unexplained_A),
            "twofold_A": gap - (explained_A + unexplained_B),
            "threefold": gap - (explained_B + unexplained_B + interaction),
            "reference": (explained_A - explained_B) - interaction,
        },
        "theorems": [
            "Research.P15.twofold_B",
            "Research.P15.twofold_A",
            "Research.P15.threefold",
            "Research.P15.reference_dependence",
            "Research.P15.explained_eq_iff",
            "Research.P15.attribution_shift",
        ],
    }


def dfl_reweight(group, x, y, weights=None) -> dict:
    """DiNardo-Fortin-Lemieux reweighting: composition and structure without a linear model.

    ``Research.P15Reweight.reweighting_matches`` / ``reweighted_mass`` / ``counterfactual_outcome`` / ``decomposition``.

    Examples
    --------
    >>> g = [True] * 6 + [False] * 6
    >>> x = ["a", "a", "a", "a", "b", "b",  "a", "a", "b", "b", "b", "b"]
    >>> y = [10, 12, 11, 13, 20, 22,  8, 9, 15, 16, 14, 17]
    >>> r = dfl_reweight(g, x, y)
    >>> tuple(round(r[k], 12) for k in ("mean_1", "mean_0", "counterfactual", "structure", "composition"))
    (14.666666666667, 13.166666666667, 10.833333333333, 3.833333333333, -2.333333333333)
    >>> (r["psi"]["x"], r["psi"]["psi"], r["max_composition_gap"])
    (['a', 'b'], [2.0, 0.5], 0.0)
    """
    n = len(group)
    if len(x) != n or len(y) != n:
        raise ValueError("group, x and y must have equal length")
    if any(v is None for v in group) or any(v is None for v in x):
        raise ValueError("no missing values allowed")
    g = []
    for v in group:
        if isinstance(v, bool):
            g.append(v)
        elif v in (0, 1):
            g.append(bool(v))
        else:
            raise ValueError("group must be logical or 0/1")
    yy = [float(v) for v in y]
    if any(math.isnan(v) for v in yy):
        raise ValueError("no missing values allowed")
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    if len(w) != n or any(math.isnan(v) or v < 0 for v in w):
        raise ValueError("weights must be non-negative")
    xs = [str(v) for v in x]
    if math.fsum(w[i] for i in range(n) if g[i]) <= 0 or math.fsum(w[i] for i in range(n) if not g[i]) <= 0:
        raise ValueError("both groups need positive total weight")
    vals = list(dict.fromkeys(xs))
    m1 = {v: math.fsum(w[i] for i in range(n) if g[i] and xs[i] == v) for v in vals}
    m0 = {v: math.fsum(w[i] for i in range(n) if not g[i] and xs[i] == v) for v in vals}
    bad = [v for v in vals if m1[v] > 0 and m0[v] == 0]
    if bad:
        raise ValueError("common support fails at x = " + ", ".join(bad))
    psi = {v: (m1[v] / m0[v] if m0[v] > 0 else 0.0) for v in vals}
    pw = [w[i] * psi[xs[i]] for i in range(n)]
    i1 = [i for i in range(n) if g[i]]
    i0 = [i for i in range(n) if not g[i]]
    mean_1 = math.fsum(w[i] * yy[i] for i in i1) / math.fsum(w[i] for i in i1)
    mean_0 = math.fsum(w[i] * yy[i] for i in i0) / math.fsum(w[i] for i in i0)
    rw_mass = math.fsum(pw[i] for i in i0)
    cf = math.fsum(pw[i] * yy[i] for i in i0) / rw_mass
    tot1 = math.fsum(m1.values())
    gap = max(abs(math.fsum(pw[i] for i in i0 if xs[i] == v) / rw_mass - m1[v] / tot1) for v in vals)
    return {
        "mean_1": mean_1,
        "mean_0": mean_0,
        "counterfactual": cf,
        "structure": mean_1 - cf,
        "composition": cf - mean_0,
        "psi": {
            "x": vals,
            "mass_1": [m1[v] for v in vals],
            "mass_0": [m0[v] for v in vals],
            "psi": [psi[v] for v in vals],
        },
        "reweighted_mass": rw_mass,
        "mass_1": tot1,
        "max_composition_gap": gap,
        "theorems": [
            "Research.P15Reweight.reweighting_matches",
            "Research.P15Reweight.reweighted_mass",
            "Research.P15Reweight.counterfactual_outcome",
            "Research.P15Reweight.decomposition",
        ],
    }
