"""Quasi-Biennial Oscillation."""

__all__ = ["qbo"]


def qbo(U30):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Quasi-Biennial Oscillation

    Formula: 30 hPa zonal wind at equator

    Parameters
    ----------
    U30 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Reed et al (1961)
    """
    raise NotImplementedError(
        "morie.fn.qboIdx.qbo is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "qboIdx: Quasi-Biennial Oscillation"
