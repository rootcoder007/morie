# morie.fn -- function file (rootcoder007/morie)
"""2D Voronoi diagram"""


def voronoi_2d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    2D Voronoi diagram

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.msvor.voronoi_2d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


voro = voronoi_2d


def cheatsheet() -> str:
    return "voronoi_2d({}) -> 2D Voronoi diagram"


# compact alias per ledger/NAMING.md
voronoi2d = voronoi_2d
