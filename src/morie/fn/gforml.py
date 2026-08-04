# morie.fn -- function file (rootcoder007/morie)
"""Robins g-formula -- Monte Carlo simulation of counterfactual outcome distribution."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["robins_g_formula"]


def _ols(X, y):
    D = np.column_stack([np.ones(X.shape[0]), X])
    b, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ b
    sd = float(np.sqrt(np.maximum((resid**2).mean(), 0.0)))
    return b, sd


def _pred(b, X):
    return np.column_stack([np.ones(X.shape[0]), X]) @ b


def robins_g_formula(y, treatment_history, covariate_history, intervention, n_mc=2000, seed=0):
    r"""Parametric g-formula by Monte Carlo for a static regime.

    Fits linear models for each time-varying confounder given the past,
    :math:`L_t \mid \bar A_{t-1}, \bar L_{t-1}`, and for the outcome
    given the full history, then simulates the counterfactual world in
    which treatment is *set* to the regime ``intervention``:

    1. draw baseline :math:`L_1^*` by resampling the observed baseline;
    2. for t = 2..T draw
       :math:`L_t^* \sim N(\hat E[L_t \mid \bar a^*, \bar L^*], \hat\sigma_t^2)`;
    3. average the fitted outcome mean over the simulated histories,

    .. math:: E[Y(\bar a)] = \int E[Y \mid \bar a, \bar l]
              \prod_t dF(l_t \mid \bar a_{t-1}, \bar l_{t-1}).

    This is exactly the substitution the naive regression cannot make
    when :math:`L_t` is both a confounder for :math:`A_t` and an effect
    of :math:`A_{t-1}`.

    Parameters
    ----------
    y : array-like, shape (n,)
        End-of-follow-up outcome.
    treatment_history : array-like of {0, 1}, shape (n, T) or (n,)
        Observed treatments.
    covariate_history : array-like, shape (n, T) or (n,)
        Time-varying confounder.
    intervention : array-like, shape (T,) or scalar
        Static regime: the treatment value forced at each period.
    n_mc : int, default 2000
        Monte Carlo sample size.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    RichResult
        keys: ``estimate`` (E[Y(abar)]), ``regime``, ``n``,
        ``n_periods``, ``n_mc``, ``method``.

    References
    ----------
    Robins, J. M. (1986). A new approach to causal inference in
    mortality studies with a sustained exposure period. *Mathematical
    Modelling*, 7, 1393-1512.

    Hernan, M. A. & Robins, J. M. (2020). *Causal Inference: What If*.
    Chapman & Hall/CRC. Ch. 21 (the parametric g-formula).
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(treatment_history, dtype=float)
    L = np.asarray(covariate_history, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if L.ndim == 1:
        L = L[:, None]
    n, T = A.shape
    if y.size != n or L.shape != (n, T):
        raise ValueError(f"shapes disagree: y {y.size}, A {A.shape}, L {L.shape}.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("treatment_history must be binary 0/1.")
    regime = np.broadcast_to(np.asarray(intervention, dtype=float).ravel(), (T,)).copy()

    # fit confounder models L_t | Abar_{t-1}, Lbar_{t-1} for t >= 2
    l_models = []
    for t in range(1, T):
        X = np.column_stack([A[:, :t], L[:, :t]])
        l_models.append(_ols(X, L[:, t]))
    # outcome model Y | Abar_T, Lbar_T
    b_y, _ = _ols(np.column_stack([A, L]), y)

    rng = np.random.default_rng(seed)
    Ls = np.empty((n_mc, T))
    Ls[:, 0] = rng.choice(L[:, 0], size=n_mc, replace=True)
    As = np.tile(regime, (n_mc, 1))
    for t in range(1, T):
        b, sd = l_models[t - 1]
        mu = _pred(b, np.column_stack([As[:, :t], Ls[:, :t]]))
        Ls[:, t] = mu + rng.normal(scale=sd, size=n_mc)
    est = float(np.mean(_pred(b_y, np.column_stack([As, Ls]))))

    return RichResult(
        payload={
            "estimate": est,
            "regime": regime,
            "n": int(n),
            "n_periods": int(T),
            "n_mc": int(n_mc),
            "method": "Robins g-formula (Monte Carlo, static regime)",
        }
    )


def cheatsheet():
    return "gforml: MC parametric g-formula E[Y(abar)] under a static regime"


# compact alias per ledger/NAMING.md
robinsgformula = robins_g_formula
