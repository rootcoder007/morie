"""
Groundwater quality index

Category: WtrQual
"""


def wqgwi(data=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Groundwater quality index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.wqgwi.wqgwi is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "wqgwi"
alias = "wqgwi"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
wqgwi = wqgwi


def cheatsheet() -> str:
    return "wqgwi({}) -> Groundwater quality index"
