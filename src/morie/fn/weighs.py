"""
Spatial sampling weights

Category: GeoProcss
"""


def weighs(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial sampling weights

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.weighs.weighs is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "weighs"
alias = "weighs"
quote = "What is now proved was once only imagined. -- William Blake"
weighs = weighs


def cheatsheet() -> str:
    return "weighs({}) -> Spatial sampling weights"
