"""Hurst exponent via R/S analysis."""

__all__ = ["hurst_exponent"]


def hurst_exponent(y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Hurst exponent via R/S analysis

    Formula: log(R/S) ~ H log(n)

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hurst (1951)
    """
    raise NotImplementedError(
        "morie.fn.hurste.hurst_exponent is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hurste: Hurst exponent via R/S analysis"


# compact alias per ledger/NAMING.md
hurstexponent = hurst_exponent
