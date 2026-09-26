"""Clark-Evans aggregation index."""

__all__ = ["clark_evans"]


def clark_evans(coords):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Clark-Evans aggregation index

    Formula: R = mean(d_NN) / E[d_NN] under CSR

    Parameters
    ----------
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Clark-Evans (1954)
    """
    raise NotImplementedError(
        "morie.fn.clstpp.clark_evans is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "clstpp: Clark-Evans aggregation index"


# compact alias per ledger/NAMING.md
clarkevans = clark_evans
