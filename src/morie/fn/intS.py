"""Symbolic indefinite integral."""

__all__ = ["symbolic_integrate"]


def symbolic_integrate(expr, x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Symbolic indefinite integral

    Formula: hybrid Risch + heuristics + table

    Parameters
    ----------
    expr : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bronstein (1997)
    """
    raise NotImplementedError(
        "morie.fn.intS.symbolic_integrate is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "intS: Symbolic indefinite integral"
