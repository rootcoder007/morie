"""Radiation model (mobility)"""


def radiation_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Radiation model (mobility)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zerad.radiation_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


radi = radiation_model


def cheatsheet() -> str:
    return "radiation_model({}) -> Radiation model (mobility)"


# compact alias per ledger/NAMING.md
radiationmodel = radiation_model
