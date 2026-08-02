"""
Machine Learning sensitivity diagnostics.
Wraps SMOTE oversampling and random forest bounds calculations.
"""

from __future__ import annotations

from typing import Any

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

class _MissingDep:
    """Placeholder for a dependency being nativized (task #141)."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportError(
            "%s is no longer bundled; this code path awaits its native "
            "morie implementation" % self._name)

    def __call__(self, *a, **k):
        raise ImportError(
            "%s is no longer bundled; this code path awaits its native "
            "morie implementation" % self._name)

try:
    from morie.fn._ml_core import RandomForestClassifier
except ImportError:
    RandomForestClassifier = _MissingDep('RandomForestClassifier')
try:
    from morie.fn._ml_core import classification_report
except ImportError:
    classification_report = _MissingDep('classification_report')


def eval_robustness(
    X: pd.DataFrame,
    y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate the robustness and predictive accuracy of a simple Random Forest classifier.

    :param X: Training features.
    :param y: Training labels.
    :param test_X: Testing features.
    :param test_y: Testing labels.
    :return: A dictionary containing the classification report metrics.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(test_X)
    report = classification_report(test_y, preds, output_dict=True)
    assert isinstance(report, dict)
    return report


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    k_neighbors: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Apply SMOTE oversampling to balance a binary outcome.

    Returns the resampled (X, y) and a status dict with class counts
    before/after and the method used.

    Falls back to random oversampling if imbalanced-learn is not installed
    or if the minority class has fewer samples than k_neighbors.
    """
    counts_before = y.value_counts().to_dict()
    minority_count = int(y.value_counts().min())
    majority_count = int(y.value_counts().max())

    # Determine k_neighbors -- SMOTE needs at least k_neighbors minority samples
    if k_neighbors is None:
        k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

    from morie.fn.smote import apply_smote as _native_smote
    X_res, y_res, _st = _native_smote(
        X, y, random_state=random_state, k_neighbors=k_neighbors)
    method = _st["method"]

    counts_after = y_res.value_counts().to_dict()
    status = {
        "method": method,
        "minority_before": minority_count,
        "majority_before": majority_count,
        "imbalance_ratio_before": round(minority_count / majority_count, 4) if majority_count > 0 else 0.0,
        "total_before": len(y),
        "total_after": len(y_res),
        **{f"class_{k}_before": v for k, v in counts_before.items()},
        **{f"class_{k}_after": v for k, v in counts_after.items()},
    }
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name), status
