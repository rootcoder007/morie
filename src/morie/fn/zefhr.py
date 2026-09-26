"""Fay-Herriot small area estimator"""


def fay_herriot(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Fay-Herriot small area estimator

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zefhr.fay_herriot is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


fay_ = fay_herriot


def cheatsheet() -> str:
    return "fay_herriot({}) -> Fay-Herriot small area estimator"


# compact alias per ledger/NAMING.md
fayherriot = fay_herriot
