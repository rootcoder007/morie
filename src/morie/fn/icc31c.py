"""ICC(1,1) one-way random effects."""

__all__ = ["icc_one_way"]


def icc_one_way(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    ICC(1,1) one-way random effects

    Formula: sigma_alpha^2 / (sigma_alpha^2 + sigma_eps^2)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shrout-Fleiss (1979)
    """
    raise NotImplementedError(
        "morie.fn.icc31c.icc_one_way is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "icc31c: ICC(1,1) one-way random effects"
