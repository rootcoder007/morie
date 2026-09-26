"""Continuous ranked probability score."""

__all__ = ["crps"]


def crps(forecast_cdf, y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Continuous ranked probability score

    Formula: CRPS(F,y) = ∫(F(z)−𝟙{z≥y})²dz

    Parameters
    ----------
    forecast_cdf : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matheson-Winkler (1976)
    """
    raise NotImplementedError(
        "morie.fn.crpsF.crps is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "crpsF: Continuous ranked probability score"
