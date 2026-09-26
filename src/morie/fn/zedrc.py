"""Spatial dose-response curve"""


def dose_resp_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial dose-response curve

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zedrc.dose_resp_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


dose = dose_resp_spatial


def cheatsheet() -> str:
    return "dose_resp_spatial({}) -> Spatial dose-response curve"
