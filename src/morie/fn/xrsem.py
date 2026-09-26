"""SEM (Spatial Error) model ML estimation"""


def sem_ml(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    SEM (Spatial Error) model ML estimation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrsem.sem_ml is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sem_ = sem_ml


def cheatsheet() -> str:
    return "sem_ml({}) -> SEM (Spatial Error) model ML estimation"
