"""Detrended fluctuation analysis (DFA)."""

__all__ = ["detrended_fluctuation"]


def detrended_fluctuation(y, scales):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Detrended fluctuation analysis (DFA)

    Formula: power-law scaling of detrended cumulative

    Parameters
    ----------
    y : array-like
        Input data.
    scales : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Peng et al (1994)
    """
    raise NotImplementedError(
        "morie.fn.detfdt.detrended_fluctuation is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "detfdt: Detrended fluctuation analysis (DFA)"
