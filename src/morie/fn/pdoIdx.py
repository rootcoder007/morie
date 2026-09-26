"""Pacific Decadal Oscillation."""

__all__ = ["pdo"]


def pdo(sst):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Pacific Decadal Oscillation

    Formula: PC1 of N. Pacific SST anomaly

    Parameters
    ----------
    sst : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mantua-Hare (2002)
    """
    raise NotImplementedError(
        "morie.fn.pdoIdx.pdo is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "pdoIdx: Pacific Decadal Oscillation"
