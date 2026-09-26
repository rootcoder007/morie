"""Gravity spatial interaction"""


def gravity_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Gravity spatial interaction

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgrv.gravity_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


grav = gravity_model


def cheatsheet() -> str:
    return "gravity_model({}) -> Gravity spatial interaction"


# compact alias per ledger/NAMING.md
gravitymodel = gravity_model
