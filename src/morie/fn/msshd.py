# morie.fn -- function file (rootcoder007/morie)
"""Shepard disparities"""


def shepard_dist(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Shepard disparities

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msshd.shepard_dist is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


shep = shepard_dist


def cheatsheet() -> str:
    return "shepard_dist({}) -> Shepard disparities"


# compact alias per ledger/NAMING.md
sheparddist = shepard_dist
