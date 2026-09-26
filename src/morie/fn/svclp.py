"""Cutting plane in 3D"""


def cut_plane(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Cutting plane in 3D

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svclp.cut_plane is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cut_ = cut_plane


def cheatsheet() -> str:
    return "cut_plane({}) -> Cutting plane in 3D"


# compact alias per ledger/NAMING.md
cutplane = cut_plane
