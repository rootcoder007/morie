"""Sample compositions from a Dirichlet distribution."""

__all__ = ["dirichlet_sample"]


def dirichlet_sample(alpha, n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Sample compositions from a Dirichlet distribution

    Formula: g_i ~ Gamma(α_i,1); x = C(g)

    Parameters
    ----------
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X

    References
    ----------
    Wilks (1962)
    """
    raise NotImplementedError(
        "morie.fn.aitdrs.dirichlet_sample is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "aitdrs: Sample compositions from a Dirichlet distribution"
