"""MGWR variable bandwidths"""


def mgwr_bandwidths(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    MGWR variable bandwidths

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrmgb.mgwr_bandwidths is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mgwr = mgwr_bandwidths


def cheatsheet() -> str:
    return "mgwr_bandwidths({}) -> MGWR variable bandwidths"


# compact alias per ledger/NAMING.md
mgwrbandwidths = mgwr_bandwidths
