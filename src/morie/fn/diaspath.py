"""Network diameter."""

__all__ = ["diameter"]


def diameter(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Network diameter

    Formula: max d(u,v)

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
    Newman (2010)
    """
    raise NotImplementedError(
        "morie.fn.diaspath.diameter is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "diaspath: Network diameter"
