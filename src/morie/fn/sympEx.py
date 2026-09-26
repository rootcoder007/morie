"""Algebraic expansion."""

__all__ = ["sympy_expand"]


def sympy_expand(expr):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Algebraic expansion

    Formula: distribute products and powers

    Parameters
    ----------
    expr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    SymPy
    """
    raise NotImplementedError(
        "morie.fn.sympEx.sympy_expand is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sympEx: Algebraic expansion"


# compact alias per ledger/NAMING.md
sympyexpand = sympy_expand
