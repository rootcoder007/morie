# morie.fn -- function file (rootcoder007/morie)
"""
Reproject coordinates between CRS

Category: GeoProcss
"""


def reproj(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Reproject coordinates between CRS

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.reproj.reproj is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "reproj"
alias = "reproj"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
reproj = reproj


def cheatsheet() -> str:
    return "reproj({}) -> Reproject coordinates between CRS"
