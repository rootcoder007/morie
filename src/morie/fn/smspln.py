"""Smoothing spline."""

__all__ = ["smoothing_spline"]


def smoothing_spline(x, y, lam):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Smoothing spline

    Formula: min sum (y_i − f(x_i))² + λ ∫ (f''(x))²

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wahba (1990)
    """
    raise NotImplementedError(
        "morie.fn.smspln.smoothing_spline is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "smspln: Smoothing spline"
