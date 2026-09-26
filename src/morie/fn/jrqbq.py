"""Jarque-Bera normality on residuals."""

__all__ = ["jarque_bera"]


def jarque_bera(resid):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Jarque-Bera normality on residuals

    Formula: JB = n/6 (S² + (K−3)²/4)

    Parameters
    ----------
    resid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jarque-Bera (1980)
    """
    raise NotImplementedError(
        "morie.fn.jrqbq.jarque_bera is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "jrqbq: Jarque-Bera normality on residuals"
