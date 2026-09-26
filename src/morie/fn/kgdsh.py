# morie.fn -- function file (rootcoder007/morie)
"""Disjunctive kriging Hermite polynomials"""


def dk_hermite(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Disjunctive kriging Hermite polynomials

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgdsh.dk_hermite is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


dk_h = dk_hermite


def cheatsheet() -> str:
    return "dk_hermite({}) -> Disjunctive kriging Hermite polynomials"


# compact alias per ledger/NAMING.md
dkhermite = dk_hermite
