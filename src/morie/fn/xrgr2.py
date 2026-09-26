"""Poisson gravity model"""


def gravity_poisson(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Poisson gravity model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgr2.gravity_poisson is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


grav = gravity_poisson


def cheatsheet() -> str:
    return "gravity_poisson({}) -> Poisson gravity model"


# compact alias per ledger/NAMING.md
gravitypoisson = gravity_poisson
