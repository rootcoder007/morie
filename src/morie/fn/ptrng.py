# morie.fn -- function file (rootcoder007/morie)
"""Point pattern intensity"""


def pp_intensity(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Point pattern intensity

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptrng.pp_intensity is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pp_i = pp_intensity


def cheatsheet() -> str:
    return "pp_intensity({}) -> Point pattern intensity"


# compact alias per ledger/NAMING.md
ppintensity = pp_intensity
