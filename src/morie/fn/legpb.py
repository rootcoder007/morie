"""Legendre polynomial basis."""

__all__ = ["legendre_basis"]


def legendre_basis(x, K):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Legendre polynomial basis

    Formula: recurrence relation

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Legendre (1782)
    """
    raise NotImplementedError(
        "morie.fn.legpb.legendre_basis is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "legpb: Legendre polynomial basis"


# compact alias per ledger/NAMING.md
legendrebasis = legendre_basis
