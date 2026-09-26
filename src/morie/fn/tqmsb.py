"""Theoretical MSE distortion bound for TurboQuant at b bits."""

__all__ = ["turboquant_mse_distortion_bound"]


def turboquant_mse_distortion_bound(bits):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Theoretical MSE distortion bound for TurboQuant at b bits

    Formula: MSE <= c_b * sigma^2;  c_b = Panter-Dite constant for b-bit Lloyd-Max

    Parameters
    ----------
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    TurboQuant MORIE integration -- mse_distortion_bound
    """
    raise NotImplementedError(
        "morie.fn.tqmsb.turboquant_mse_distortion_bound is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "tqmsb: Theoretical MSE distortion bound for TurboQuant at b bits"
