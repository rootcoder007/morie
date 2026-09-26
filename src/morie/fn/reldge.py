"""Reliability of GEBV."""

__all__ = ["reliability_gebv"]


def reliability_gebv(fit):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Reliability of GEBV

    Formula: r^2 = 1 - PEV/sigma_a^2

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Henderson (1984)
    """
    raise NotImplementedError(
        "morie.fn.reldge.reliability_gebv is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "reldge: Reliability of GEBV"
