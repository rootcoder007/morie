# morie.fn -- function file (rootcoder007/morie)
"""Inhomogeneous Poisson process"""


def inhom_poisson(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Inhomogeneous Poisson process

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptipo.inhom_poisson is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


inho = inhom_poisson


def cheatsheet() -> str:
    return "inhom_poisson({}) -> Inhomogeneous Poisson process"


# compact alias per ledger/NAMING.md
inhompoisson = inhom_poisson
