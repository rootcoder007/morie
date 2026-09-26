"""Network transitivity."""

__all__ = ["transitivity"]


def transitivity(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Network transitivity

    Formula: 3 * triangles / triads

    Parameters
    ----------
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman (2003)
    """
    raise NotImplementedError(
        "morie.fn.trnscl.transitivity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "trnscl: Network transitivity"
