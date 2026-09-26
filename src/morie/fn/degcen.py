"""Degree centrality."""

__all__ = ["degree_centrality"]


def degree_centrality(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Degree centrality

    Formula: C_D(v) = deg(v) / (n-1)

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
    Freeman (1979)
    """
    raise NotImplementedError(
        "morie.fn.degcen.degree_centrality is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "degcen: Degree centrality"
