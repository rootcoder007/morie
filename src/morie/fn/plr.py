# morie.fn -- function file (rootcoder007/morie)
"""Partially Linear Regression (PLR): native double machine learning.

Cross-fitted ridge nuisances and the partialling-out orthogonal score,
computed entirely on morie's native cores. No external DML library.
"""

from __future__ import annotations

from . import _array_core as np
from . import _frame_core as pd
from ._ml_core import RidgeCV
from ._stats_core import norm as _norm


def _folds(n, k, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i::k] for i in range(k)]


def estimate_plr(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    r"""
    Partially Linear Regression (PLR) ATE via native double ML.

    The PLR model is:

    .. math::

        Y = \theta_0 D + g_0(X) + \varepsilon, \quad
        D = m_0(X) + v

    Nuisances :math:`\ell_0(X) = \mathbb{E}[Y\mid X]` and
    :math:`m_0(X) = \mathbb{E}[D\mid X]` are ridge regressions,
    cross-fitted over ``n_folds`` folds. With residuals
    :math:`\hat u = Y - \hat\ell` and :math:`\hat v = D - \hat m`,
    the partialling-out score :math:`\psi = (\hat u - \theta\hat v)\hat v`
    gives

    .. math::

        \hat\theta = \frac{\sum_i \hat v_i \hat u_i}{\sum_i \hat v_i^2},
        \qquad
        \widehat{se}^2 = \frac{n^{-1}\sum_i \psi_i^2}
                              {(n^{-1}\sum_i \hat v_i^2)^2 \, n}

    (Chernozhukov et al. 2018, sec. 4.1, "partialling out"; this is
    also the score DoubleML's ``DoubleMLPLR`` uses by default).

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the treatment variable.
    :param outcome: Column name of the outcome variable.
    :param covariates: List of covariate column names.
    :param n_folds: Number of cross-fitting folds. Default 5.
    :param random_state: Random seed for the fold split. Default 42.
    :return: dict with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Chernozhukov et al. (2018). Double/debiased machine learning for
        treatment and structural parameters. Econometrics Journal,
        21(1), C1-C68.
    """
    required_cols = [treatment, outcome] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    df = data[[treatment, outcome] + covariates].dropna()
    X = [[float(df[c].tolist()[i]) for c in covariates]
         for i in range(len(df))]
    y = [float(v) for v in df[outcome].tolist()]
    d = [float(v) for v in df[treatment].tolist()]
    n_obs = len(y)

    rng = np.random.default_rng(random_state)
    lhat = [0.0] * n_obs
    mhat = [0.0] * n_obs
    for fold in _folds(n_obs, n_folds, rng):
        train = [i for i in range(n_obs) if i not in set(fold)]
        Xtr = [X[i] for i in train]
        ml_l = RidgeCV().fit(Xtr, [y[i] for i in train])
        ml_m = RidgeCV().fit(Xtr, [d[i] for i in train])
        Xf = [X[i] for i in fold]
        pl = ml_l.predict(Xf)
        pm = ml_m.predict(Xf)
        pl = pl.tolist() if hasattr(pl, "tolist") else list(pl)
        pm = pm.tolist() if hasattr(pm, "tolist") else list(pm)
        for j, i in enumerate(fold):
            lhat[i] = float(pl[j])
            mhat[i] = float(pm[j])

    u = [y[i] - lhat[i] for i in range(n_obs)]
    v = [d[i] - mhat[i] for i in range(n_obs)]
    vv = sum(x * x for x in v)
    if vv == 0.0:
        raise ValueError("treatment residual variance is zero; the "
                         "treatment is fully explained by covariates")
    ate = sum(a * b for a, b in zip(v, u)) / vv
    psi = [(u[i] - ate * v[i]) * v[i] for i in range(n_obs)]
    j0 = vv / n_obs
    se = ((sum(p * p for p in psi) / n_obs) / (j0 * j0) / n_obs) ** 0.5
    z = ate / se if se > 0 else float("inf")
    pval = 2.0 * float(_norm.sf(abs(z)))
    zc = 1.959963984540054

    return {
        "ate": float(ate),
        "se": float(se),
        "ci_lower": float(ate - zc * se),
        "ci_upper": float(ate + zc * se),
        "pval": pval,
        "n_obs": n_obs,
    }


plr_fn = estimate_plr


def cheatsheet() -> str:
    return ("estimate_plr({}) -> Native double-ML PLR: cross-fitted "
            "ridge nuisances, partialling-out score.")


# compact alias per ledger/NAMING.md
estimateplr = estimate_plr
