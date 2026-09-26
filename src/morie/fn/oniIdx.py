"""Oceanic Niño Index."""

__all__ = ["oni"]


def oni(sst_n34):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Oceanic Niño Index

    Formula: 3-month running mean Niño 3.4 anomaly

    Parameters
    ----------
    sst_n34 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    NOAA CPC
    """
    raise NotImplementedError(
        "morie.fn.oniIdx.oni is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "oniIdx: Oceanic Niño Index"
