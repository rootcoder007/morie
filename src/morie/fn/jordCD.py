"""Jordan canonical form."""

__all__ = ["jordan_canonical"]


def jordan_canonical(A):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Jordan canonical form

    Formula: P J P^{-1}; J block-diagonal

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jordan (1870)
    """
    raise NotImplementedError(
        "morie.fn.jordCD.jordan_canonical is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "jordCD: Jordan canonical form"
