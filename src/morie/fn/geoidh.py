# morie.fn -- function file (rootcoder007/morie)
"""
Geoid height computation

Category: GeoProcss
"""


def geoidh(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Geoid height computation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.geoidh.geoidh is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "geoidh"
alias = "geoidh"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
geoidh = geoidh


def cheatsheet() -> str:
    return "geoidh({}) -> Geoid height computation"
