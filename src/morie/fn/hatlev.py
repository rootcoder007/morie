"""Hat-matrix leverage h_ii."""

__all__ = ["leverage"]


def leverage(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Hat-matrix leverage h_ii

    Formula: H = X(X^T X)^{-1}X^T; h_ii = diag(H)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hoaglin-Welsch (1978)
    """
    raise NotImplementedError(
        "morie.fn.hatlev.leverage is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hatlev: Hat-matrix leverage h_ii"
