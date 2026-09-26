"""Betweenness centrality."""

__all__ = ["betweenness_centrality"]


def betweenness_centrality(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Betweenness centrality

    Formula: C_B(v) = sum sigma(s,t|v)/sigma(s,t)

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
    Freeman (1977); Brandes (2001)
    """
    raise NotImplementedError(
        "morie.fn.betcen.betweenness_centrality is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "betcen: Betweenness centrality"
