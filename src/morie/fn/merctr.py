# morie.fn -- function file (rootcoder007/morie)
"""
Mercator projection

Category: GeoProcss
"""


def merctr(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Mercator projection

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.merctr.merctr is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "merctr"
alias = "merctr"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
merctr = merctr


def cheatsheet() -> str:
    return "merctr({}) -> Mercator projection"
