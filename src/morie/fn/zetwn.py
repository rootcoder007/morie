"""Townsend deprivation index"""


def townsend_index(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Townsend deprivation index

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zetwn.townsend_index is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


town = townsend_index


def cheatsheet() -> str:
    return "townsend_index({}) -> Townsend deprivation index"


# compact alias per ledger/NAMING.md
townsendindex = townsend_index
