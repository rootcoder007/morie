"""Symbolic limit."""

__all__ = ["symbolic_limit"]


def symbolic_limit(expr, x, x0):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Symbolic limit

    Formula: L'Hôpital + series + Gruntz

    Parameters
    ----------
    expr : array-like
        Input data.
    x : array-like
        Input data.
    x0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gruntz (1996)
    """
    raise NotImplementedError(
        "morie.fn.limT.symbolic_limit is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "limT: Symbolic limit"


# compact alias per ledger/NAMING.md
symboliclimit = symbolic_limit
