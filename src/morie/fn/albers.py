# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""
Albers equal area projection

Category: GeoProcss
"""


def albers(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Albers equal area projection

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.albers.albers is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "albers"
alias = "albers"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
albers = albers


def cheatsheet() -> str:
    return "albers({}) -> Albers equal area projection"
