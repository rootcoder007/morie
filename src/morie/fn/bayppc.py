"""Posterior predictive check."""

__all__ = ["posterior_predictive_check"]


def posterior_predictive_check(y, y_rep, statistic):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Posterior predictive check

    Formula: discrepancy T(y, theta) vs T(y_rep, theta)

    Parameters
    ----------
    y : array-like
        Input data.
    y_rep : array-like
        Input data.
    statistic : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman-Meng-Stern (1996)
    """
    raise NotImplementedError(
        "morie.fn.bayppc.posterior_predictive_check is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayppc: Posterior predictive check"
