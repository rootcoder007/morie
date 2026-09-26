"""Spatial Durbin model (lagged covariates)."""

__all__ = ["spatial_durbin_model"]


def spatial_durbin_model(y, X, W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Spatial Durbin model (lagged covariates)

    Formula: y = rho W y + X beta + W X theta + epsilon

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
    LeSage & Pace (2009)
    """
    raise NotImplementedError(
        "morie.fn.sdurbm.spatial_durbin_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sdurbm: Spatial Durbin model (lagged covariates)"
