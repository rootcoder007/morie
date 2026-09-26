"""Hierarchical spatial (nested)"""


def hier_spatial_fe(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Hierarchical spatial (nested)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxhrs.hier_spatial_fe is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


hier = hier_spatial_fe


def cheatsheet() -> str:
    return "hier_spatial_fe({}) -> Hierarchical spatial (nested)"


# compact alias per ledger/NAMING.md
hierspatialfe = hier_spatial_fe
