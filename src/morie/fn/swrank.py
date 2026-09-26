"""Rank of spatial weights matrix."""


def swrank(W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Rank of spatial weights matrix.

    Category: WDiag

    Parameters
    ----------
    W : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.swrank.swrank is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


swrank_fn = swrank


def cheatsheet() -> str:
    return "swrank({}) -> Rank of spatial weights matrix."
