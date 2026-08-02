# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""IPW-weighted OLS ATE estimator."""

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
    import statsmodels.formula.api as smf
except ImportError:
    smf = _MissingDep('smf')


def estimate_ate(data: pd.DataFrame, outcome: str, treatment: str, weights_col: str) -> tuple[float, float]:
    """
    Estimate Average Treatment Effect (ATE) using a weighted linear model.

    :param data: The pandas DataFrame containing the analytical sample.
    :type data: pandas.DataFrame
    :param outcome: The name of the outcome variable column.
    :type outcome: str
    :param treatment: The name of the binary treatment indicator column.
    :type treatment: str
    :param weights_col: The name of the column containing the analytical weights (e.g. IPTW).
    :type weights_col: str
    :return: A tuple containing the estimated ATE coefficient and its standard error.
    :rtype: tuple[float, float]
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
