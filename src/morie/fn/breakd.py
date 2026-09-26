"""Finite-sample breakdown point."""

__all__ = ["breakdown_point"]


def breakdown_point(estimator, n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Finite-sample breakdown point

    Formula: smallest fraction of contamination that ruins estimator

    Parameters
    ----------
    estimator : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Donoho-Huber (1983)
    """
    raise NotImplementedError(
        "morie.fn.breakd.breakdown_point is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "breakd: Finite-sample breakdown point"


# compact alias per ledger/NAMING.md
breakdownpoint = breakdown_point
