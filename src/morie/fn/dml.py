# morie.fn -- function file (rootcoder007/morie)
"""
Double Machine Learning (DML) -- Partially Linear Regression Model.

Implements ``estimate_double_ml`` -- estimates the ATE using DoubleML's PLR
with Random Forest nuisance estimators and cross-fitting.
"""

from __future__ import annotations

from . import _array_core as np
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
    from ._ml_core import RandomForestClassifier, RandomForestRegressor
except ImportError:
    RandomForestClassifier = _MissingDep('RandomForestClassifier')
    RandomForestRegressor = _MissingDep('RandomForestRegressor')

_DOUBLEML_RANDOM_STATE: int = 42
"""Module-level seed for all DoubleML estimations.  Change at call-site if needed."""

_DOUBLEML_N_FOLDS: int = 5
"""Number of cross-fitting folds.  Must be >= 2."""

_DOUBLEML_N_REP: int = 1
"""Number of repeated cross-fitting repetitions.  Set > 1 to reduce variance."""


def estimate_double_ml(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list,
    *,
    random_state: int = _DOUBLEML_RANDOM_STATE,
    n_folds: int = _DOUBLEML_N_FOLDS,
    n_rep: int = _DOUBLEML_N_REP,
):
    """
    Estimate the Average Treatment Effect using Double Machine Learning (DML).

    Uses :class:`doubleml.DoubleMLPLR` (Partially Linear Regression Model)
    with Random Forest nuisance estimators for both the outcome regression
    (``ml_l``) and the treatment model (``ml_m``).

    Reproducibility
    ---------------
    All stochastic operations are seeded deterministically:

    1. ``numpy`` global seed is set to ``random_state`` immediately before
       constructing the learners.
    2. Both ``RandomForestRegressor`` instances receive ``random_state``.
    3. The ``DoubleMLPLR`` object is constructed with ``n_folds`` and
       ``n_rep`` passed explicitly so that the cross-fitting schedule is
       fixed for a given seed.

    To change the seed for a sensitivity run, pass ``random_state=<int>``.

    :param data: The dataset containing outcome, treatment, and covariates.
    :type data: pandas.DataFrame
    :param outcome: Name of the continuous outcome variable.
    :type outcome: str
    :param treatment: Name of the treatment variable.
    :type treatment: str
    :param covariates: List of covariate column names to control for.
    :type covariates: list[str]
    :param random_state: Integer seed for all RNGs.  Default 42.
    :type random_state: int, optional
    :param n_folds: Number of cross-fitting folds.  Default 5.
    :type n_folds: int, optional
    :param n_rep: Number of repeated cross-fitting repetitions.  Default 1.
    :type n_rep: int, optional
    :return: dict with ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs`` from the native cross-fitted PLR.
    :rtype: dict

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., & Robins, J. (2018). Double/debiased machine learning for
    treatment and structural parameters. *The Econometrics Journal*, 21(1),
    C1--C68. https://doi.org/10.1111/ectj.12097
    """
    del n_rep  # single-rep native estimator; kept for signature compat
    from .plr import estimate_plr as _native_plr
    return _native_plr(data, treatment=treatment, outcome=outcome,
                       covariates=covariates, n_folds=n_folds,
                       random_state=random_state)


dml = estimate_double_ml


def cheatsheet() -> str:
    return "estimate_double_ml({}) -> Double Machine Learning (DML) -- Partially Linear Regression "
