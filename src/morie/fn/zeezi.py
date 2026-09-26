"""Ecological zero-inflated"""


def ecological_zip(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ecological zero-inflated

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zeezi.ecological_zip is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


ecol = ecological_zip


def cheatsheet() -> str:
    return "ecological_zip({}) -> Ecological zero-inflated"


# compact alias per ledger/NAMING.md
ecologicalzip = ecological_zip
