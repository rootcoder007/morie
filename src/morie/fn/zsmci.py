"""Monte Carlo spatial integration"""


def mc_spatial_int(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Monte Carlo spatial integration

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsmci.mc_spatial_int is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


mc_s = mc_spatial_int


def cheatsheet() -> str:
    return "mc_spatial_int({}) -> Monte Carlo spatial integration"


# compact alias per ledger/NAMING.md
mcspatialint = mc_spatial_int
