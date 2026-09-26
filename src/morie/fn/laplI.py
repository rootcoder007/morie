"""Inverse Laplace transform."""

__all__ = ["inverse_laplace"]


def inverse_laplace(F, s, t):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Inverse Laplace transform

    Formula: Bromwich integral / partial-fraction

    Parameters
    ----------
    F : array-like
        Input data.
    s : array-like
        Input data.
    t : array-like
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
        "morie.fn.laplI.inverse_laplace is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "laplI: Inverse Laplace transform"


# compact alias per ledger/NAMING.md
inverselaplace = inverse_laplace
