"""Indirect standardization"""


def indirect_std(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Indirect standardization

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zeism.indirect_std is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


indi = indirect_std


def cheatsheet() -> str:
    return "indirect_std({}) -> Indirect standardization"


# compact alias per ledger/NAMING.md
indirectstd = indirect_std
