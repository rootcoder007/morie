"""Infomap community via random walk MDL."""

__all__ = ["infomap"]


def infomap(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Infomap community via random walk MDL

    Formula: min description length of random walk

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
    Rosvall-Bergstrom (2008)
    """
    raise NotImplementedError(
        "morie.fn.cominf.infomap is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "cominf: Infomap community via random walk MDL"
