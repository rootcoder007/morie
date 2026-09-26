"""DSSP secondary structure assignment."""

__all__ = ["dssp_secondary"]


def dssp_secondary(coords):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    DSSP secondary structure assignment

    Formula: hydrogen-bond pattern -> 8-state alphabet

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
    Kabsch-Sander (1983)
    """
    raise NotImplementedError(
        "morie.fn.dssprt.dssp_secondary is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "dssprt: DSSP secondary structure assignment"


# compact alias per ledger/NAMING.md
dsspsecondary = dssp_secondary
