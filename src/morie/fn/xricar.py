"""Intrinsic CAR (ICAR)"""


def icar_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Intrinsic CAR (ICAR)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xricar.icar_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


icar = icar_model


def cheatsheet() -> str:
    return "icar_model({}) -> Intrinsic CAR (ICAR)"


# compact alias per ledger/NAMING.md
icarmodel = icar_model
