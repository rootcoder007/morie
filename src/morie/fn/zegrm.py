"""Gravity migration model"""


def gravity_migration(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Gravity migration model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zegrm.gravity_migration is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


grav = gravity_migration


def cheatsheet() -> str:
    return "gravity_migration({}) -> Gravity migration model"
