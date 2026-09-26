"""Turning bands simulation"""


def turning_bands(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Turning bands simulation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zstbs.turning_bands is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


turn = turning_bands


def cheatsheet() -> str:
    return "turning_bands({}) -> Turning bands simulation"


# compact alias per ledger/NAMING.md
turningbands = turning_bands
