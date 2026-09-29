# morie.fn -- function file (rootcoder007/morie)
"""G-computation (outcome regression) ATE estimator with bootstrap SE."""

import warnings

from . import _array_core as np
from . import _frame_core as pd


from ._ml_core import LinearRegression, LogisticRegression
from ._ml_core import StandardScaler
from ._rng import random_uniform


def estimate_ate_gcomputation(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    outcome_model: str = "linear",
) -> dict:
    r"""G-computation (outcome regression / standardisation) ATE estimator.

    1. Fit an outcome model ``E[Y | T, X]`` on the complete cases: ordinary
       least squares (``"linear"``) or unpenalised maximum-likelihood
       logistic regression (``"logistic"``), both with an intercept and
       main effects of ``T`` and ``X``.
    2. Predict ``Y(1)`` and ``Y(0)`` for every unit by setting ``T = 1``
       and ``T = 0``.
    3. The ATE is ``n^{-1} sum_i (Y_i(1) - Y_i(0))`` (Robins 1986; Hernan
       and Robins 2020, ch. 13).

    The standard error is the standard deviation of 500 nonparametric
    bootstrap replicates of the whole procedure (replicate ``b`` resamples
    rows with Philox stream ``b`` of seed 42, so the R twin ``GComp``
    draws the same rows; resamples holding a single treatment arm are
    skipped), and the interval is their 2.5% and 97.5% type-7 percentiles.

    Parameters
    ----------
    data : DataFrame
        Input frame (at least 10 complete rows).
    treatment, outcome : str
        Binary treatment and outcome columns.
    covariates : list of str
        Confounders.
    outcome_model : {"linear", "logistic"}
        Outcome regression.

    Returns
    -------
    dict
        ``ate``, ``se``, ``ci_lower``, ``ci_upper``, ``n_obs``,
        ``outcome_model``.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in mortality studies with a sustained
    exposure period. *Mathematical Modelling*, 7, 1393-1512.

    Hernan, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC, ch. 13.

    Examples
    --------
    >>> d = pd.DataFrame({"t": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    ...                   "x": [0.2, 1.1, -0.5, 0.9, 1.4, 0.1, 0.3, -1.0, 2.0, 0.6, -0.2, 0.8],
    ...                   "y": [1.1, 3.4, 0.2, 3.0, 3.9, 1.0, 2.5, -0.3, 4.6, 1.6, 2.0, 1.9]})
    >>> r = estimate_ate_gcomputation(d, treatment="t", outcome="y", covariates=["x"])
    >>> round(r["ate"], 10), round(r["se"], 10)
    (1.2485390115, 0.0709248273)
    """
    valid_models = {"linear", "logistic"}
    if outcome_model not in valid_models:
        raise ValueError(f"outcome_model must be one of {valid_models}.")

    required_cols = [treatment, outcome] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")

    df = data[[treatment, outcome] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)
    if n_obs < 10:
        raise ValueError("G-computation requires at least 10 complete observations.")

    feature_cols = [treatment] + covariates

    def _fit_and_predict_ate(df_boot: pd.DataFrame) -> float:
        X = df_boot[feature_cols].astype(float).values
        y = df_boot[outcome].astype(float).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if outcome_model == "linear":
            model = LinearRegression()
        else:
            model = LogisticRegression(penalty=None, max_iter=500)

        model.fit(X_scaled, y)

        # Counterfactual datasets: all treated / all control
        X_t1 = df_boot[feature_cols].astype(float).copy()
        X_t0 = df_boot[feature_cols].astype(float).copy()
        X_t1[treatment] = 1.0
        X_t0[treatment] = 0.0

        X_t1_scaled = scaler.transform(X_t1.values)
        X_t0_scaled = scaler.transform(X_t0.values)

        if outcome_model == "linear":
            y1_hat = model.predict(X_t1_scaled)
            y0_hat = model.predict(X_t0_scaled)
        else:
            y1_hat = model.predict_proba(X_t1_scaled)[:, 1]
            y0_hat = model.predict_proba(X_t0_scaled)[:, 1]

        return float(np.mean(y1_hat - y0_hat))

    # Point estimate
    ate = _fit_and_predict_ate(df)

    # Bootstrap SE: replicate b resamples rows with Philox stream b of seed 42
    boot_ates = []
    tvals = [float(v) for v in df[treatment].tolist()]
    for b in range(500):
        u = random_uniform(n_obs, seed=42, stream=b)
        idx = [min(int(float(v) * n_obs), n_obs - 1) for v in (u.tolist() if hasattr(u, "tolist") else u)]
        if len({tvals[i] for i in idx}) < 2:
            continue  # one arm only: the treatment effect is not identified in this resample
        boot_df = df.iloc[idx].reset_index(drop=True)
        try:
            boot_ates.append(_fit_and_predict_ate(boot_df))
        except Exception:
            continue

    if len(boot_ates) < 50:
        warnings.warn(
            "Fewer than 50 successful bootstrap iterations; SE may be unreliable.",
            stacklevel=2,
        )

    se = float(np.std(boot_ates, ddof=1)) if len(boot_ates) > 1 else float("nan")
    ci_lower = float(np.percentile(boot_ates, 2.5)) if boot_ates else float("nan")
    ci_upper = float(np.percentile(boot_ates, 97.5)) if boot_ates else float("nan")

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "outcome_model": outcome_model,
    }


g_comp_fn = estimate_ate_gcomputation


def cheatsheet() -> str:
    return "estimate_ate_gcomputation({}) -> G-computation (outcome regression) ATE estimator with bootst"
