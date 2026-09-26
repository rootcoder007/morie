"""Girvan-Newman edge-betweenness."""

__all__ = ["girvan_newman"]


def girvan_newman(G):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Girvan-Newman edge-betweenness

    Formula: iteratively remove highest-betweenness edge

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
    Girvan-Newman (2002)
    """
    raise NotImplementedError(
        "morie.fn.comgir.girvan_newman is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "comgir: Girvan-Newman edge-betweenness"


# compact alias per ledger/NAMING.md
girvannewman = girvan_newman
