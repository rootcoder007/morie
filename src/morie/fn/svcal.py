"""Calvert uncertainty model"""


def calvert_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Calvert uncertainty model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svcal.calvert_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


calv = calvert_model


def cheatsheet() -> str:
    return "calvert_model({}) -> Calvert uncertainty model"


# compact alias per ledger/NAMING.md
calvertmodel = calvert_model
