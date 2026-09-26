"""Distance-band weights"""


def w_distance(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Distance-band weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwds.w_distance is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_di = w_distance


def cheatsheet() -> str:
    return "w_distance({}) -> Distance-band weights"


# compact alias per ledger/NAMING.md
wdistance = w_distance
