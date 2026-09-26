"""Local Getis-Ord G_i*."""

__all__ = ["local_getis_g"]


def local_getis_g(x, W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Local Getis-Ord G_i*

    Formula: G_i* = sum_j w_ij x_j / sum_j x_j

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Getis-Ord (1992); Ord-Getis (1995)
    """
    raise NotImplementedError(
        "morie.fn.lisgst.local_getis_g is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "lisgst: Local Getis-Ord G_i*"


# compact alias per ledger/NAMING.md
localgetisg = local_getis_g
