"""L1 (LAD) regression."""

__all__ = ["l1_regression"]


def l1_regression(X, y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    L1 (LAD) regression

    Formula: min sum |y_i − x_i^T β|

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bloomfield-Steiger (1983)
    """
    raise NotImplementedError(
        "morie.fn.lqsr.l1_regression is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "lqsr: L1 (LAD) regression"


# compact alias per ledger/NAMING.md
l1regression = l1_regression
