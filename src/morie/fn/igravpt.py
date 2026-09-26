# morie.fn -- function file (rootcoder007/morie)
"""Spatial potential model."""


def igravpt(mass, dist):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial potential model.

    Category: Gravity

    Parameters
    ----------
    mass, dist : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.igravpt.igravpt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


igravpt_fn = igravpt


def cheatsheet() -> str:
    return "igravpt({}) -> Spatial potential model."
