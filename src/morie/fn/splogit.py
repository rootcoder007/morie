"""Spatial logit estimation."""


def splogit(y, X, W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial logit estimation.

    Category: SProbit

    Parameters
    ----------
    y, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.splogit.splogit is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


splogit_fn = splogit


def cheatsheet() -> str:
    return "splogit({}) -> Spatial logit estimation."
