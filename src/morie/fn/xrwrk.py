"""Rook contiguity weights"""


def w_rook(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Rook contiguity weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwrk.w_rook is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_ro = w_rook


def cheatsheet() -> str:
    return "w_rook({}) -> Rook contiguity weights"


# compact alias per ledger/NAMING.md
wrook = w_rook
