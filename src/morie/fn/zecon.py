"""Concentration index spatial"""


def concentration_idx(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Concentration index spatial

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zecon.concentration_idx is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


conc = concentration_idx


def cheatsheet() -> str:
    return "concentration_idx({}) -> Concentration index spatial"
