"""Symbolic differentiation."""

__all__ = ["symbolic_diff"]


def symbolic_diff(expr, x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Symbolic differentiation

    Formula: chain + product + quotient rules

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
    classical
    """
    raise NotImplementedError(
        "morie.fn.diffs.symbolic_diff is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "diffS: Symbolic differentiation"


# compact alias per ledger/NAMING.md
symbolicdiff = symbolic_diff
