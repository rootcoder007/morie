"""PolarQuant radius+angle decomposition of a d-dim vector."""

__all__ = ["turboquant_polar_transform"]


def turboquant_polar_transform(x):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    PolarQuant radius+angle decomposition of a d-dim vector

    Formula: r = ||x||_2;  theta_i = atan2(x_{i+1}, x_i)  for successive pairs;  x = (r, theta)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r, theta

    References
    ----------
    TurboQuant MORIE integration -- morie/quant.py polar_transform
    """
    raise NotImplementedError(
        "morie.fn.tqpol.turboquant_polar_transform is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "tqpol: PolarQuant radius+angle decomposition of a d-dim vector"
