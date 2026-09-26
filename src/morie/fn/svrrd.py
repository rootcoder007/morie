"""Roemer party unanimity model"""


def roemer_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Roemer party unanimity model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svrrd.roemer_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


roem = roemer_model


def cheatsheet() -> str:
    return "roemer_model({}) -> Roemer party unanimity model"


# compact alias per ledger/NAMING.md
roemermodel = roemer_model
