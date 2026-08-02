# morie.fn -- function file (rootcoder007/morie)
"""Random Forest classifier robustness evaluation."""

from __future__ import annotations

from typing import Any

from . import _frame_core as pd

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
    from ._ml_core import RandomForestClassifier
except ImportError:
    RandomForestClassifier = _MissingDep('RandomForestClassifier')
try:
    from ._ml_core import classification_report
except ImportError:
    classification_report = _MissingDep('classification_report')


def eval_robustness(
    X: pd.DataFrame,
    y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> dict[str, Any]:
    """Evaluate predictive accuracy of a Random Forest classifier.

    Parameters
    ----------
    X : DataFrame
        Training features.
    y : Series
        Training labels.
    test_X : DataFrame
        Testing features.
    test_y : Series
        Testing labels.

    Returns
    -------
    dict
        Classification report metrics (precision, recall, f1, support).
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(test_X)
    report = classification_report(test_y, preds, output_dict=True)
    assert isinstance(report, dict)
    return report


robust = eval_robustness


def cheatsheet() -> str:
    return "eval_robustness({}) -> Random Forest classifier robustness evaluation."
