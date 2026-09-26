"""Systematic sample with random start."""

__all__ = ["systematic_sample"]


def systematic_sample(frame, n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Systematic sample with random start

    Formula: every k-th element after random offset

    Parameters
    ----------
    frame : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Madow-Madow (1944)
    """
    raise NotImplementedError(
        "morie.fn.sysmpl.systematic_sample is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sysmpl: Systematic sample with random start"
