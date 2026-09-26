"""
Soil structure index

Category: SoilSp
"""


def sostr(data=None, depth=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Soil structure index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.sostr.sostr is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "sostr"
alias = "sostr"
quote = "The measure of a man is what he does with power. -- Plato"
sostr = sostr


def cheatsheet() -> str:
    return "sostr({}) -> Soil structure index"
