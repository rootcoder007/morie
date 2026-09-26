"""Inverse distance weights matrix"""


def w_inverse_dist(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Inverse distance weights matrix

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwid.w_inverse_dist is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_in = w_inverse_dist


def cheatsheet() -> str:
    return "w_inverse_dist({}) -> Inverse distance weights matrix"


# compact alias per ledger/NAMING.md
winversedist = w_inverse_dist
