# morie.fn -- function file (rootcoder007/morie)
"""2D Delaunay triangulation"""


def delaunay_2d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    2D Delaunay triangulation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msdel.delaunay_2d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


dela = delaunay_2d


def cheatsheet() -> str:
    return "delaunay_2d({}) -> 2D Delaunay triangulation"


# compact alias per ledger/NAMING.md
delaunay2d = delaunay_2d
