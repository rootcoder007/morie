# morie.fn -- function file (rootcoder007/morie)
"""
Mollweide projection

Category: GeoProcss
"""


def mollwd(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Mollweide projection

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.mollwd.mollwd is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "mollwd"
alias = "mollwd"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
mollwd = mollwd


def cheatsheet() -> str:
    return "mollwd({}) -> Mollweide projection"
