"""Spectral radius of W."""


def swspec(W):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spectral radius of W.

    Category: WDiag

    Parameters
    ----------
    W : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.swspec.swspec is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


swspec_fn = swspec


def cheatsheet() -> str:
    return "swspec({}) -> Spectral radius of W."
