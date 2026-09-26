"""Double-weighting (calibration + nonresponse)."""

__all__ = ["rake_double_weights"]


def rake_double_weights(w_nr, w_cal):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Double-weighting (calibration + nonresponse)

    Formula: product of nonresponse + calibration weights

    Parameters
    ----------
    w_nr : array-like
        Input data.
    w_cal : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Särndal-Lundström (2005)
    """
    raise NotImplementedError(
        "morie.fn.rakedw.rake_double_weights is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "rakedw: Double-weighting (calibration + nonresponse)"
