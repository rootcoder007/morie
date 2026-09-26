# morie.fn -- function file (rootcoder007/morie)
"""
Satellite ocean color

Category: MarinSp
"""


def masat(depth=None, temp=None, salinity=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Satellite ocean color

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.masat.masat is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "masat"
alias = "masat"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
masat = masat


def cheatsheet() -> str:
    return "masat({}) -> Satellite ocean color"
