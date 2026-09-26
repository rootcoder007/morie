"""GP predictive variance"""


def gp_variance(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GP predictive variance

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsgpv.gp_variance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gp_v = gp_variance


def cheatsheet() -> str:
    return "gp_variance({}) -> GP predictive variance"
