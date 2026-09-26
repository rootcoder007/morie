# morie.fn -- function file (rootcoder007/morie)
"""Cross-type point pattern"""


def cross_pp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Cross-type point pattern

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptcrs.cross_pp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cros = cross_pp


def cheatsheet() -> str:
    return "cross_pp({}) -> Cross-type point pattern"


# compact alias per ledger/NAMING.md
crosspp = cross_pp
