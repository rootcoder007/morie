# SPDX-License-Identifier: AGPL-3.0-or-later
"""Research P15: sentencing disparity decompositions (``research/lean/P15Decomposition.lean``; Oaxaca 1973; Blinder 1973).

* ``Research.P15.twofold_B`` / ``twofold_A`` / ``threefold``
* ``Research.P15.reference_dependence`` / ``explained_eq_iff`` / ``attribution_shift``

R parity: ``rmorie`` ``R/disparity_decomposition.R`` (``morie_disparity_decomposition``).
"""

from __future__ import annotations

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

__all__ = ["disparity_decomposition"]


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
