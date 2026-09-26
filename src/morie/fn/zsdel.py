"""Delaunay triangulation mesh"""


def delaunay_mesh(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Delaunay triangulation mesh

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsdel.delaunay_mesh is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


dela = delaunay_mesh


def cheatsheet() -> str:
    return "delaunay_mesh({}) -> Delaunay triangulation mesh"


# compact alias per ledger/NAMING.md
delaunaymesh = delaunay_mesh
