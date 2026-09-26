# morie.fn -- function file (rootcoder007/morie)
"""
Sea surface chlorophyll

Category: MarinSp
"""


def massc(depth=None, temp=None, salinity=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Sea surface chlorophyll

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.massc.massc is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "massc"
alias = "massc"
quote = "There is no royal road to geometry. -- Euclid"
massc = massc


def cheatsheet() -> str:
    return "massc({}) -> Sea surface chlorophyll"
