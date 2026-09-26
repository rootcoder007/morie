# morie.fn -- function file (rootcoder007/morie)
"""Gaussian random field with Matern covariance."""


def rfgss(data=None, coords=None, n=100, seed=42, **kwargs):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Gaussian random field with Matern covariance

    Parameters
    ----------
    data : array-like, optional
        Observed values at sample locations.
    coords : array-like, optional
        Coordinates of sample locations, shape (n, 2) or (n, 3).
    n : int
        Number of simulation nodes or grid points (default 100).
    seed : int
        Random seed for reproducibility (default 42).
    **kwargs
        Additional method-specific parameters.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.rfgss.rfgss is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


rfgss = rfgss


def cheatsheet() -> str:
    return "rfgss({}) -> Gaussian random field with Matern covariance."
