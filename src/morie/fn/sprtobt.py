"""Spatial Tobit model."""


def sprtobt(y, X, W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial Tobit model.

    Category: SProbit

    Parameters
    ----------
    y, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.sprtobt.sprtobt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sprtobt_fn = sprtobt


def cheatsheet() -> str:
    return "sprtobt({}) -> Spatial Tobit model."
