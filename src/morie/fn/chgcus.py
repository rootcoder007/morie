"""CUSUM changepoint detection."""

__all__ = ["changepoint_cusum"]


def changepoint_cusum(y, k, h):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CUSUM changepoint detection

    Formula: S_t = max(0, S_{t-1} + (x_t - mu_0 - k))

    Parameters
    ----------
    y : array-like
        Input data.
    k : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Page (1954)
    """
    raise NotImplementedError(
        "morie.fn.chgcus.changepoint_cusum is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "chgcus: CUSUM changepoint detection"
