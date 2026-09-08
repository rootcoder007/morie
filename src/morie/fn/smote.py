# morie.fn -- function file (rootcoder007/morie)
"""SMOTE oversampling, implemented natively.

Chawla, Bowyer, Hall & Kegelmeyer (2002), "SMOTE: Synthetic Minority
Over-sampling Technique", *JAIR* 16, 321-357, sec. 4: for each needed
synthetic sample, pick a minority point, pick one of its k nearest
minority neighbours, and interpolate a uniform fraction of the way
along the joining segment. Index and gap draws follow imbalanced-learn's
``BaseSMOTE._make_samples`` (random base pick with replacement across
the flattened neighbour table; step ~ U[0, 1)), so behaviour matches
what the previous imblearn code path produced.
"""

from __future__ import annotations

from typing import Any

from . import _array_core as np
from . import _frame_core as pd


def _knn_indices(pts: list[list[float]], k: int) -> list[list[int]]:
    """Indices of each point's k nearest neighbours (self excluded)."""
    out = []
    for i, a in enumerate(pts):
        d = []
        for j, b in enumerate(pts):
            if j == i:
                continue
            d.append((sum((x - y) ** 2 for x, y in zip(a, b)), j))
        d.sort()
        out.append([j for _, j in d[:k]])
    return out


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    k_neighbors: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Apply SMOTE oversampling to balance a binary outcome.

    Synthetic minority samples are interpolated between real minority
    points and their nearest minority neighbours until the classes
    balance. When the minority class is too small even for one
    neighbour (n < 2), plain random oversampling with replacement is
    used and reported as such.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (numeric).
    y : Series
        Binary outcome.
    random_state : int
        Random seed (default 42).
    k_neighbors : int or None
        SMOTE neighbour count. Auto-selected if None
        (min(5, minority - 1), imblearn's effective default).

    Returns
    -------
    tuple[DataFrame, Series, dict]
        Resampled (X, y) and a status dict with class counts
        before/after and the method used.
    """
    counts_before = y.value_counts().to_dict()
    minority_count = int(y.value_counts().min())
    majority_count = int(y.value_counts().max())
    minority_label = y.value_counts().idxmin()

    if k_neighbors is None:
        k_neighbors = min(5, minority_count - 1) \
            if minority_count > 1 else 1

    cols = list(X.columns)
    rows = [[float(X[c].tolist()[i]) for c in cols]
            for i in range(len(y))]
    labels = list(y.tolist())
    n_needed = majority_count - minority_count
    rng = np.random.default_rng(random_state)

    new_rows: list[list[float]] = []
    if n_needed > 0 and minority_count >= 2:
        method = "smote"
        m_idx = [i for i, v in enumerate(labels) if v == minority_label]
        pts = [rows[i] for i in m_idx]
        nn = _knn_indices(pts, min(k_neighbors, len(pts) - 1))
        for _ in range(n_needed):
            base = int(rng.integers(0, len(pts)))
            nb = pts[nn[base][int(rng.integers(0, len(nn[base])))]]
            gap = float(rng.uniform(0.0, 1.0))
            new_rows.append([a + gap * (b - a)
                             for a, b in zip(pts[base], nb)])
    elif n_needed > 0 and minority_count == 1:
        method = "random_oversample"
        seed_row = rows[labels.index(minority_label)]
        new_rows = [list(seed_row) for _ in range(n_needed)]
    else:
        method = "none"

    all_rows = rows + new_rows
    all_labels = labels + [minority_label] * len(new_rows)
    X_res = pd.DataFrame({c: [r[j] for r in all_rows]
                          for j, c in enumerate(cols)})
    y_res = pd.Series(all_labels, name=getattr(y, "name", None))
    counts_after = y_res.value_counts().to_dict()

    return X_res, y_res, {
        "method": method,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "n_synthetic": len(new_rows),
        "k_neighbors": k_neighbors,
    }


# compact alias per ledger/NAMING.md
applysmote = apply_smote
