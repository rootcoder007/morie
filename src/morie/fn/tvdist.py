"""Total variation distance."""

__all__ = ["total_variation_distance"]


def total_variation_distance(y, p, q):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Total variation distance

    Formula: TV(P,Q) = (1/2) sum_x |p(x) - q(x)|

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Devroye & Gyorfi (1985)
    """
    raise NotImplementedError(
        "morie.fn.tvdist.total_variation_distance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "tvdist: Total variation distance"
