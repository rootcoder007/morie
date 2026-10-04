# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""IPW-weighted OLS ATE estimator."""

from . import _frame_core as pd
from ._glm_core import formula as smf


def estimate_ate(data: pd.DataFrame, outcome: str, treatment: str, weights_col: str) -> tuple[float, float]:
    r"""IPW average treatment effect by weighted least squares with HC3 errors.

    Regresses the outcome on an intercept and the binary treatment with
    analytic weights ``w`` (e.g. ``1/e(X)`` for the treated and ``1/(1 -
    e(X))`` for controls); the treatment coefficient is then the Hajek
    contrast ``sum_T w y / sum_T w - sum_C w y / sum_C w`` (Hernan and
    Robins 2020, ch. 12). Its standard error is the HC3 sandwich of the
    weighted fit (MacKinnon and White 1985), computed on the rows scaled
    by ``sqrt(w)``: ``(X'WX)^{-1} X'W diag(e_i^2 / (1 - h_ii)^2) W X
    (X'WX)^{-1}`` with ``h_ii`` the weighted hat values. The propensity
    model is taken as known, so the SE ignores its estimation.

    Parameters
    ----------
    data : DataFrame
        Analytical sample.
    outcome, treatment, weights_col : str
        Outcome, binary treatment and weight columns.

    Returns
    -------
    tuple of float
        ``(ate, se)``.

    References
    ----------
    Hernan, M. A. and Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC, ch. 12.

    MacKinnon, J. G. and White, H. (1985). Some heteroskedasticity-consistent covariance matrix
    estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325.

    Examples
    --------
    >>> d = pd.DataFrame({"y": [1.0, 2.2, 1.7, 3.1, 2.8, 3.9], "t": [0, 0, 0, 1, 1, 1],
    ...                   "w": [1.2, 2.0, 1.5, 1.1, 3.0, 1.4]})
    >>> [round(v, 10) for v in estimate_ate(d, outcome="y", treatment="t", weights_col="w")]
    [1.4059574468, 0.6454726427]
    """
    formula = f"{outcome} ~ {treatment}"
    # HC3 robust covariance: corrects for heteroskedasticity introduced by
    # unequal IPTW weights.  Plain OLS/WLS SEs are downward-biased when
    # observation weights vary widely, producing anti-conservative inference.
    model = smf.wls(formula=formula, data=data, weights=data[weights_col]).fit(cov_type="HC3")
    return float(model.params[treatment]), float(model.bse[treatment])


ate_fn = estimate_ate


def cheatsheet() -> str:
    return "estimate_ate({}) -> IPW-weighted OLS ATE estimator."


# compact alias per ledger/NAMING.md
estimateate = estimate_ate
