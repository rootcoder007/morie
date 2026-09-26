# morie.fn -- function file (rootcoder007/morie)
"""Co-kriging multivariate"""


def cokriging(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Co-kriging multivariate

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgcok.cokriging is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cokr = cokriging


def cheatsheet() -> str:
    return "cokriging({}) -> Co-kriging multivariate"
