"""Symbolic matrix algebra."""

__all__ = ["matrix_symbolic"]


def matrix_symbolic(M):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Symbolic matrix algebra

    Formula: det, inv, eigenvalues over symbols

    Parameters
    ----------
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical linear algebra
    """
    raise NotImplementedError(
        "morie.fn.matSym.matrix_symbolic is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "matSym: Symbolic matrix algebra"


# compact alias per ledger/NAMING.md
matrixsymbolic = matrix_symbolic
