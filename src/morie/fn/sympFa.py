"""Polynomial factorization."""

__all__ = ["sympy_factor"]


def sympy_factor(expr):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Polynomial factorization

    Formula: Berlekamp-Zassenhaus / Cantor-Zassenhaus

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
    Cohen (1996) book
    """
    raise NotImplementedError(
        "morie.fn.sympFa.sympy_factor is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sympFa: Polynomial factorization"


# compact alias per ledger/NAMING.md
sympyfactor = sympy_factor
