"""Spatial gradient estimation"""


def gradient_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial gradient estimation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zegrn.gradient_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


grad = gradient_spatial


def cheatsheet() -> str:
    return "gradient_spatial({}) -> Spatial gradient estimation"
