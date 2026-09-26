"""Label propagation community detection."""

__all__ = ["label_propagation"]


def label_propagation(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Label propagation community detection

    Formula: each node adopts most-common neighbor label

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
    Raghavan-Albert-Kumara (2007)
    """
    raise NotImplementedError(
        "morie.fn.comlpa.label_propagation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "comlpa: Label propagation community detection"
