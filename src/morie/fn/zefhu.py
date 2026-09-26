"""Unit-level Fay-Herriot"""


def fay_herriot_unit(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Unit-level Fay-Herriot

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zefhu.fay_herriot_unit is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


fay_ = fay_herriot_unit


def cheatsheet() -> str:
    return "fay_herriot_unit({}) -> Unit-level Fay-Herriot"


# compact alias per ledger/NAMING.md
fayherriotunit = fay_herriot_unit
