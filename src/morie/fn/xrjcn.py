"""Join count statistic"""


def join_count(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Join count statistic

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrjcn.join_count is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


join = join_count


def cheatsheet() -> str:
    return "join_count({}) -> Join count statistic"
