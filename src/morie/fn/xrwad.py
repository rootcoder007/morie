"""Adaptive bandwidth weights"""


def w_adaptive(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Adaptive bandwidth weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwad.w_adaptive is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_ad = w_adaptive


def cheatsheet() -> str:
    return "w_adaptive({}) -> Adaptive bandwidth weights"


# compact alias per ledger/NAMING.md
wadaptive = w_adaptive
