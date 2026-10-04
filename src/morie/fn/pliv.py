# morie.fn -- function file (rootcoder007/morie)
"""Partially linear IV model by double/debiased machine learning (native)."""

from . import _frame_core as pd
from ._rng import random_uniform


def estimate_pliv(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    r"""Partially linear IV model by double/debiased machine learning.

    Model ``Y = theta D + g(X) + e``, ``Z = m(X) + v`` with ``E[e | Z, X] =
    0`` (Chernozhukov et al. 2018, sec. 4.2). The nuisances ``l(X) = E[Y |
    X]``, ``m(X) = E[Z | X]`` and ``r(X) = E[D | X]`` are fitted by ridge
    regression with the penalty chosen from ``(0.1, 1, 10)`` by leave-one-out
    PRESS, cross-fitted over ``n_folds`` folds (without covariates, the
    fold-complement means). With ``u = Y - l``, ``w = Z - m``, ``v = D - r``
    the partialling-out score ``psi = (u - theta v) w`` gives ``theta =
    sum(w u) / sum(w v)`` and ``se = sqrt(mean(psi^2) / J^2 / n)``, ``J =
    mean(w v)``. Rows are assigned to folds by ordering Philox uniforms
    (seed ``random_state``) and dealing them out in turn, so the R twin
    ``Pliv`` builds the same folds.

    Parameters
    ----------
    data : DataFrame
        Input frame.
    treatment, outcome, instrument : str
        Endogenous treatment, outcome and instrument columns.
    covariates : list of str
        Exogenous covariates.
    n_folds : int
        Cross-fitting folds.
    random_state : int
        Philox seed of the fold assignment.

    Returns
    -------
    dict
        ``late``, ``se``, ``ci_lower``, ``ci_upper``, ``pval``, ``n_obs``,
        ``method``.

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J.
    (2018). Double/debiased machine learning for treatment and structural parameters.
    *Econometrics Journal*, 21(1), C1-C68.

    Examples
    --------
    >>> d = pd.DataFrame({"z": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    ...                   "d": [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    ...                   "x": [0.3, -0.2, 1.1, 0.4, -0.9, 0.0, 0.7, 1.5, -0.4, 0.2, -1.2, 0.9],
    ...                   "y": [0.5, 2.1, 1.4, 2.6, 0.1, 0.2, 2.9, 3.1, 1.9, 0.6, -0.8, 3.0]})
    >>> r = estimate_pliv(d, treatment="d", outcome="y", instrument="z", covariates=["x"], n_folds=3)
    >>> round(r["late"], 10), round(r["se"], 10)
    (2.3239954858, 0.3623974801)
    """
    required_cols = [treatment, outcome, instrument] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")

    df = data[[treatment, outcome, instrument] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)

    from ._ml_core import RidgeCV
    from ._stats_core import norm as _norm

    X = (
        [[float(df[c].tolist()[i]) for c in covariates] for i in range(n_obs)]
        if covariates
        else [[] for _ in range(n_obs)]
    )
    y = [float(v) for v in df[outcome].tolist()]
    d = [float(v) for v in df[treatment].tolist()]
    z = [float(v) for v in df[instrument].tolist()]

    u_ = random_uniform(n_obs, seed=random_state)
    u_ = [float(v) for v in (u_.tolist() if hasattr(u_, "tolist") else u_)]
    idx = sorted(range(n_obs), key=lambda i: (u_[i], i))
    folds = [idx[i::n_folds] for i in range(n_folds)]

    lhat = [0.0] * n_obs
    mhat = [0.0] * n_obs
    rhat = [0.0] * n_obs
    for fold in folds:
        train = [i for i in range(n_obs) if i not in set(fold)]
        Xtr = [X[i] for i in train]
        if covariates:
            ml_l = RidgeCV().fit(Xtr, [y[i] for i in train])
            ml_m = RidgeCV().fit(Xtr, [z[i] for i in train])
            ml_r = RidgeCV().fit(Xtr, [d[i] for i in train])
            Xf = [X[i] for i in fold]
            pl, pm, pr = (ml_l.predict(Xf), ml_m.predict(Xf), ml_r.predict(Xf))
            pl = pl.tolist() if hasattr(pl, "tolist") else list(pl)
            pm = pm.tolist() if hasattr(pm, "tolist") else list(pm)
            pr = pr.tolist() if hasattr(pr, "tolist") else list(pr)
        else:
            # no covariates: the conditional means are the fold-
            # complement sample means
            pl = [sum(y[i] for i in train) / len(train)] * len(fold)
            pm = [sum(z[i] for i in train) / len(train)] * len(fold)
            pr = [sum(d[i] for i in train) / len(train)] * len(fold)
        for j, i in enumerate(fold):
            lhat[i] = float(pl[j])
            mhat[i] = float(pm[j])
            rhat[i] = float(pr[j])

    # IV-type orthogonal score (Chernozhukov et al. 2018, sec. 4.2):
    # with u = Y - l(X), w = Z - m(X), v = D - r(X),
    # psi = (u - theta v) w, so theta = E[wu]/E[wv].
    u = [y[i] - lhat[i] for i in range(n_obs)]
    w = [z[i] - mhat[i] for i in range(n_obs)]
    v = [d[i] - rhat[i] for i in range(n_obs)]
    wv = sum(a * b for a, b in zip(w, v))
    if wv == 0.0:
        raise ValueError(
            "instrument residual is orthogonal to the "
            "treatment residual; the instrument carries "
            "no identifying variation"
        )
    late = sum(a * b for a, b in zip(w, u)) / wv
    psi = [(u[i] - late * v[i]) * w[i] for i in range(n_obs)]
    j0 = wv / n_obs
    se = ((sum(p_ * p_ for p_ in psi) / n_obs) / (j0 * j0) / n_obs) ** 0.5
    zstat = late / se if se > 0 else float("inf")
    pval = 2.0 * float(_norm.sf(abs(zstat)))
    zc = 1.959963984540054
    ci_lower = late - zc * se
    ci_upper = late + zc * se
    method = "PLIV (native DML, cross-fitted ridge nuisances)"

    return {
        "late": late,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pval": pval,
        "n_obs": n_obs,
        "method": method,
    }


pliv_fn = estimate_pliv


def cheatsheet() -> str:
    return "estimate_pliv({}) -> Partially Linear IV (PLIV) for LATE via DoubleML or 2SLS fal"


# compact alias per ledger/NAMING.md
estimatepliv = estimate_pliv
