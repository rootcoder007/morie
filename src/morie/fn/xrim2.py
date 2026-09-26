"""SDM impacts decomposition"""


def sdm_impacts(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    SDM impacts decomposition

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrim2.sdm_impacts is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sdm_ = sdm_impacts


def cheatsheet() -> str:
    return "sdm_impacts({}) -> SDM impacts decomposition"


# compact alias per ledger/NAMING.md
sdmimpacts = sdm_impacts
