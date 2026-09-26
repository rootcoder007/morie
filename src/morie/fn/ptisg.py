# morie.fn -- function file (rootcoder007/morie)
"""Isotropic edge correction"""


def isotropic_guard(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Isotropic edge correction

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptisg.isotropic_guard is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


isot = isotropic_guard


def cheatsheet() -> str:
    return "isotropic_guard({}) -> Isotropic edge correction"


# compact alias per ledger/NAMING.md
isotropicguard = isotropic_guard
