# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""
Interactive Regression Model (IRM): native double machine learning.

Implements ``estimate_irm`` -- the ATE with treatment-effect
heterogeneity via Neyman-orthogonal scores and K-fold cross-fitting,
computed entirely on morie's native cores (random-forest nuisances from
``_ml_core``, RNG from ``_array_core``). No external DML library.
"""

from __future__ import annotations

from typing import Any

from . import _array_core as np
from . import _frame_core as pd
from ._ml_core import (
    LabelEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
)

# DoubleML truncates propensity scores at this threshold by default
# (trimming_rule="truncate", trimming_threshold=1e-2); the orthogonal
# score divides by m and 1-m, so unbounded propensities explode it.
_TRIM = 1e-2


def _folds(n, k, rng):
    """K roughly equal disjoint folds over shuffled indices."""
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i::k] for i in range(k)]


def estimate_irm(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    r"""Estimate the ATE via the Interactive Regression Model (IRM).

    The IRM extends the partially linear model by allowing treatment
    effect heterogeneity.  It models:

    .. math::

        Y = g_0(T, X) + U, \quad \mathbb{E}[U \mid X, T] = 0

        T = m_0(X) + V, \quad \mathbb{E}[V \mid X] = 0

    where :math:`g_0` is the outcome regression and :math:`m_0` the
    propensity score.  The Neyman-orthogonal score for the ATE is

    .. math::

        \psi = g_0(1, X) - g_0(0, X)
        + \frac{T(Y - g_0(1,X))}{m_0(X)}
        - \frac{(1-T)(Y - g_0(0,X))}{1 - m_0(X)} - \theta

    (Chernozhukov et al. 2018, eq. 5.1), with the nuisances
    cross-fitted: each fold is scored by forests trained on its
    complement, so the estimate never reuses data that trained its own
    nuisance.  :math:`\hat\theta` is the mean of the score's constant
    part and the variance estimate is the sample variance of
    :math:`\psi` (theorem 3.2).

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column.
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.
    :type covariates: list[str]
    :param n_folds: Number of cross-fitting folds (default 5).
    :type n_folds: int
    :param random_state: Random seed (default 42).
    :type random_state: int
    :return: Dictionary with ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
    C., Newey, W., & Robins, J. (2018). Double/debiased machine
    learning for treatment and structural parameters. *The Econometrics
    Journal*, 21(1), C1--C68. https://doi.org/10.1111/ectj.12097
    """
    frame = data[[treatment, outcome, *covariates]].dropna().copy()

    # Encode non-numeric covariates
    for col in covariates:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            le = LabelEncoder()
            frame[col] = le.fit_transform(
                [str(v) for v in frame[col].tolist()])

    X = [[float(frame[c].tolist()[i]) for c in covariates]
         for i in range(len(frame))]
    y = [float(v) for v in frame[outcome].tolist()]
    t = [float(v) for v in frame[treatment].tolist()]
    n = len(y)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2 for cross-fitting")

    rng = np.random.default_rng(random_state)
    g0hat = [0.0] * n
    g1hat = [0.0] * n
    mhat = [0.0] * n
    for fold in _folds(n, n_folds, rng):
        train = [i for i in range(n) if i not in set(fold)]
        tr0 = [i for i in train if t[i] == 0.0]
        tr1 = [i for i in train if t[i] == 1.0]
        if not tr0 or not tr1:
            raise ValueError(
                "a cross-fitting fold has no treated or no control "
                "units; use fewer folds or more data")
        rf0 = RandomForestRegressor(n_estimators=100, max_depth=5,
                                    random_state=random_state)
        rf0.fit([X[i] for i in tr0], [y[i] for i in tr0])
        rf1 = RandomForestRegressor(n_estimators=100, max_depth=5,
                                    random_state=random_state)
        rf1.fit([X[i] for i in tr1], [y[i] for i in tr1])
        rfm = RandomForestClassifier(n_estimators=100, max_depth=5,
                                     random_state=random_state)
        rfm.fit([X[i] for i in train], [int(t[i]) for i in train])
        Xf = [X[i] for i in fold]
        p0 = rf0.predict(Xf)
        p1 = rf1.predict(Xf)
        pm = rfm.predict_proba(Xf)
        p0 = p0.tolist() if hasattr(p0, "tolist") else list(p0)
        p1 = p1.tolist() if hasattr(p1, "tolist") else list(p1)
        pm = pm.tolist() if hasattr(pm, "tolist") else list(pm)
        for j, i in enumerate(fold):
            g0hat[i] = float(p0[j])
            g1hat[i] = float(p1[j])
            prob1 = pm[j][1] if isinstance(pm[j], (list, tuple)) \
                else float(pm[j])
            mhat[i] = min(max(float(prob1), _TRIM), 1.0 - _TRIM)

    psi_b = [g1hat[i] - g0hat[i]
             + t[i] * (y[i] - g1hat[i]) / mhat[i]
             - (1.0 - t[i]) * (y[i] - g0hat[i]) / (1.0 - mhat[i])
             for i in range(n)]
    ate = sum(psi_b) / n
    psi = [v - ate for v in psi_b]
    se = (sum(v * v for v in psi) / n / n) ** 0.5
    z = 1.959964

    return {
        "ate": float(ate),
        "se": float(se),
        "ci_lower": float(ate - z * se),
        "ci_upper": float(ate + z * se),
        "n": n,
        "method": "IRM (native DML, cross-fitted RF nuisances)",
    }


irm = estimate_irm


def cheatsheet() -> str:
    return ("estimate_irm({}) -> Native double-ML IRM: cross-fitted "
            "forest nuisances, Neyman-orthogonal ATE score.")


# compact alias per ledger/NAMING.md
estimateirm = estimate_irm
