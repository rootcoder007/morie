"""Condition number of W."""


def swcond(W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Condition number of W.

    Category: WDiag

    Parameters
    ----------
    W : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.swcond.swcond is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


swcond_fn = swcond


def cheatsheet() -> str:
    return "swcond({}) -> Condition number of W."
