"""Community modularity Q on partition."""

__all__ = ["community_modularity"]


def community_modularity(G, partition):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Community modularity Q on partition

    Formula: Q from given partition

    Parameters
    ----------
    G : array-like
        Input data.
    partition : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman (2006)
    """
    raise NotImplementedError(
        "morie.fn.comten.community_modularity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "comten: Community modularity Q on partition"
