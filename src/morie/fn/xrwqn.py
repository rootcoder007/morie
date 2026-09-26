"""Queen contiguity weights"""


def w_queen(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Queen contiguity weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwqn.w_queen is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_qu = w_queen


def cheatsheet() -> str:
    return "w_queen({}) -> Queen contiguity weights"


# compact alias per ledger/NAMING.md
wqueen = w_queen
