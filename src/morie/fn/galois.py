"""Galois group of polynomial."""

__all__ = ["galois_group"]


def galois_group(poly):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Galois group of polynomial

    Formula: compute via resolvents / Stauduhar

    Parameters
    ----------
    poly : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Galois (1832)
    """
    raise NotImplementedError(
        "morie.fn.galois.galois_group is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "galois: Galois group of polynomial"


# compact alias per ledger/NAMING.md
galoisgroup = galois_group
