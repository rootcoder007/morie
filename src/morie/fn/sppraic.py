"""Spatial probit AIC."""


def sppraic(ll, k, n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial probit AIC.

    Category: SProbit

    Parameters
    ----------
    ll, k, n : see function signature.

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.sppraic.sppraic is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sppraic_fn = sppraic


def cheatsheet() -> str:
    return "sppraic({}) -> Spatial probit AIC."
