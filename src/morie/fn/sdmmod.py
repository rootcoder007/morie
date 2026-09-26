"""Spatial Durbin model."""

__all__ = ["spatial_durbin"]


def spatial_durbin(y, X, W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Spatial Durbin model

    Formula: y = rho W y + X beta + W X theta + eps

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    LeSage-Pace (2009)
    """
    raise NotImplementedError(
        "morie.fn.sdmmod.spatial_durbin is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sdmmod: Spatial Durbin model"


# compact alias per ledger/NAMING.md
spatialdurbin = spatial_durbin
