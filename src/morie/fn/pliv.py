# morie.fn -- function file (rootcoder007/morie)
"""Partially Linear IV (PLIV) for LATE via DoubleML or 2SLS fallback."""


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
    from . import _glm_core as sm
except ImportError:
    sm = _MissingDep('sm')


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
    r"""
    Partially Linear IV (PLIV) for the Local Average Treatment Effect (LATE)
    using an instrumental variable via DoubleML.

    The PLIV model is:

    .. math::

        Y = \\theta_0 D + g_0(X) + \\varepsilon, \\quad
        D = m_0(X) + f_0(Z, X) + v

    where Z is the instrument, satisfying relevance and exclusion restriction.

    Falls back to two-stage least squares (2SLS) via statsmodels if
    DoubleML is unavailable.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the endogenous treatment variable.
    :param outcome: Column name of the outcome variable.
    :param instrument: Column name of the instrument.
    :param covariates: List of exogenous covariate column names.
    :param n_folds: Cross-fitting folds (DoubleML path). Default 5.
    :param random_state: Random seed. Default 42.
    :return: dict with keys ``late``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``, ``method``.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Chernozhukov et al. (2018). Double/debiased machine learning.
        Econometrics Journal, 21(1), C1-C68.
    Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of
        causal effects using instrumental variables. JASA, 91(434), 444-455.
    """
    required_cols = [treatment, outcome, instrument] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")

    df = data[[treatment, outcome, instrument] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)

    from ._ml_core import RidgeCV
    from ._stats_core import norm as _norm

    X = [[float(df[c].tolist()[i]) for c in covariates]
         for i in range(n_obs)] if covariates else [[] for _ in range(n_obs)]
    y = [float(v) for v in df[outcome].tolist()]
    d = [float(v) for v in df[treatment].tolist()]
    z = [float(v) for v in df[instrument].tolist()]

    rng = np.random.default_rng(random_state)
    idx = list(range(n_obs))
    rng.shuffle(idx)
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
            pl, pm, pr = (ml_l.predict(Xf), ml_m.predict(Xf),
                          ml_r.predict(Xf))
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
        raise ValueError("instrument residual is orthogonal to the "
                         "treatment residual; the instrument carries "
                         "no identifying variation")
    late = sum(a * b for a, b in zip(w, u)) / wv
    psi = [(u[i] - late * v[i]) * w[i] for i in range(n_obs)]
    j0 = wv / n_obs
    se = ((sum(p_ * p_ for p_ in psi) / n_obs) / (j0 * j0)
          / n_obs) ** 0.5
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
