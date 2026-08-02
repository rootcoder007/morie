# morie.fn -- function file (rootcoder007/morie)
"""Ridge regression with R-style verbose result."""

from collections.abc import Sequence
from typing import Union

from . import _array_core as np

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
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = _MissingDep('Ridge')


def rdgr(
    X: Union[Sequence, np.ndarray], y: Union[Sequence, np.ndarray], alpha: float = 1.0, fit_intercept: bool = True
):
    """Ridge regression (L2-regularized OLS).

    Minimizes ||y - X.beta||^2 + alpha ||beta||^2.
    """
    from ._richresult import RichResult

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)
    coefs = model.coef_
    intercept = float(model.intercept_) if fit_intercept else 0.0
    r2 = float(model.score(X, y))
    coef_rows = [[f"x{i + 1}", f"{c:.6g}"] for i, c in enumerate(coefs)]
    return RichResult(
        title="Ridge regression",
        summary_lines=[
            ("Alpha (L2)", alpha),
            ("R^2 (training)", r2),
            ("Intercept", intercept),
            ("n predictors", len(coefs)),
            ("n observations", len(y)),
        ],
        tables=[
            {
                "title": "Coefficients:",
                "headers": ["Predictor", "Coefficient"],
                "rows": coef_rows,
            }
        ],
        warnings=[] if alpha > 0 else ["alpha=0; behaves like plain OLS."],
        payload={"coef": coefs.tolist(), "intercept": intercept, "r2": r2, "alpha": alpha},
    )
