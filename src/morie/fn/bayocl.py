"""Bayesian outlier detection via DP."""

__all__ = ["bayes_outlier_dp"]


def bayes_outlier_dp(y, alpha):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Bayesian outlier detection via DP

    Formula: posterior cluster size = 1 indicates outlier

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Quintana-Iglesias (2003)
    """
    raise NotImplementedError(
        "morie.fn.bayocl.bayes_outlier_dp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayocl: Bayesian outlier detection via DP"


# compact alias per ledger/NAMING.md
bayesoutlierdp = bayes_outlier_dp
