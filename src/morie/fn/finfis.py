"""Fisher information matrix."""

__all__ = ["fisher_information"]


def fisher_information(log_likelihood, theta):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Fisher information matrix

    Formula: I(theta) = -E[d^2 log L / d theta^2]

    Parameters
    ----------
    log_likelihood : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fisher (1922)
    """
    raise NotImplementedError(
        "morie.fn.finfis.fisher_information is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "finfis: Fisher information matrix"
