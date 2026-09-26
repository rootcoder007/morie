"""Linearization (Taylor) variance."""

__all__ = ["linearization_se"]


def linearization_se(estimator, data):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Linearization (Taylor) variance

    Formula: Var(g(theta)) ~ g'(theta) Var(theta) g'(theta)

    Parameters
    ----------
    estimator : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Binder (1983)
    """
    raise NotImplementedError(
        "morie.fn.linear.linearization_se is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "linear: Linearization (Taylor) variance"
