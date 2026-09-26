"""Wolfson bipolarization index"""


def polarization_wolf(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Wolfson bipolarization index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svplw.polarization_wolf is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pola = polarization_wolf


def cheatsheet() -> str:
    return "polarization_wolf({}) -> Wolfson bipolarization index"
