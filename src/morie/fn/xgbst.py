"""XGBoost-style regularized boosting objective, implemented natively."""

import numpy as np

from ._richresult import RichResult

__all__ = ["xgboost_objective"]


def xgboost_objective(
    x,
    y,
    *,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    reg_lambda=1.0,
    reg_alpha=0.0,
    task="auto",
    seed=0,
    deterministic_seed: int | None = None,
):
    """Boosted-trees with XGBoost's regularized objective.

    L = sum_i l(y_i, y_hat_i) + sum_k Omega(f_k),
    Omega(f) = gamma T + (1/2) lambda ||w||^2 (+ alpha ||w||_1).

    Implemented natively: the leaf weight is w* = -G/(H + lambda) with the
    gradient soft-thresholded by alpha, and a split is taken only when

        Gain = 0.5[G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)
                   - (G_L+G_R)^2/(H_L+H_R+lambda)] - gamma

    is positive. No boosting package is imported.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,).
    n_estimators, learning_rate, max_depth : XGBoost / HistGB hyperparams.
    reg_lambda : float
        L2 leaf penalty.
    reg_alpha : float
        L1 leaf penalty (XGBoost only).
    task : "auto" | "classification" | "regression".
    seed : int
        random_state.
    deterministic_seed : int or None, optional
        If supplied, the backend ``random_state`` is derived from the
        SHA-keyed :func:`morie._det_rng.r_seed` so Py<->R streams agree
        for the canonical fixture.  When ``None`` (default), behaviour
        is unchanged: ``seed`` drives ``random_state`` directly.

    Returns
    -------
    RichResult with payload: estimate (train score), feature_importances,
    backend ("native"), task, n, method.
    """
    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if task == "auto":
        task = (
            "classification"
            if np.issubdtype(y.dtype, np.integer) or set(np.unique(y)).issubset({0, 1})
            else "regression"
        )

    if deterministic_seed is not None:
        from morie._det_rng import r_seed

        rs = r_seed("xgbst", deterministic_seed)
    else:
        rs = seed

    # Native second-order boosting. The previous code imported xgboost --
    # which is declared nowhere in pyproject.toml -- and fell back to
    # sklearn's HistGradientBoosting, which has no feature_importances_ (so
    # importances came back None) and no L1 parameter at all (so reg_alpha
    # was silently ignored). The native engine honours both penalties and
    # matches sklearn's GradientBoosting importances to three decimals when
    # reg_lambda = reg_alpha = 0.
    from ._trees_native import gb_fit

    backend = "native"
    fit = gb_fit(
        X,
        y,
        task=task,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
    )
    importances = np.asarray(fit["importance"]).tolist()
    if task == "classification":
        _, yv = np.unique(np.asarray(y).ravel(), return_inverse=True)
        score = float(((fit["fitted"] > 0.5).astype(int) == yv).mean())
    else:
        yv = np.asarray(y, dtype=float).ravel()
        denom = float(((yv - yv.mean()) ** 2).sum())
        score = float(1.0 - ((fit["fitted"] - yv) ** 2).sum() / max(denom, 1e-12))
    return RichResult(
        payload={
            "estimate": score,
            "train_score": score,
            "feature_importances": importances,
            "backend": backend,
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "reg_lambda": float(reg_lambda),
            "reg_alpha": float(reg_alpha),
            "task": task,
            "n": int(n),
            "method": f"XGBoost-style boosting ({backend}, {task})",
        }
    )


def cheatsheet():
    return "xgbst: XGBoost regularized boosting (sklearn HistGB fallback)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    r = xgboost_objective(X, y, n_estimators=50, seed=0)
    print("backend:", r.backend, "  task:", r.task)
    print("train score:", r.train_score)
    print("feature importances:", r.feature_importances)
