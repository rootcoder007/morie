# morie.fn -- function file (rootcoder007/morie)
"""Point pattern Voronoi intensities"""


def pp_voronoi(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Point pattern Voronoi intensities

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptvor.pp_voronoi is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pp_v = pp_voronoi


def cheatsheet() -> str:
    return "pp_voronoi({}) -> Point pattern Voronoi intensities"


# compact alias per ledger/NAMING.md
ppvoronoi = pp_voronoi
