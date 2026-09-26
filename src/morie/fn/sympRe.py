"""SymPy simplify expression."""

__all__ = ["sympy_simplify"]


def sympy_simplify(expr):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    SymPy simplify expression

    Formula: normalize via various heuristics

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
    SymPy team (2017)
    """
    raise NotImplementedError(
        "morie.fn.sympRe.sympy_simplify is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sympRe: SymPy simplify expression"


# compact alias per ledger/NAMING.md
sympysimplify = sympy_simplify
