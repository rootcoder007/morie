"""Leroux CAR model"""


def leroux_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Leroux CAR model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zelrx.leroux_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


lero = leroux_model


def cheatsheet() -> str:
    return "leroux_model({}) -> Leroux CAR model"


# compact alias per ledger/NAMING.md
lerouxmodel = leroux_model
