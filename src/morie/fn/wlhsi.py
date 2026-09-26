"""
Habitat suitability index

Category: WildlSp
"""


def wlhsi(abundance=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Habitat suitability index

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.wlhsi.wlhsi is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "wlhsi"
alias = "wlhsi"
quote = "Number rules the universe. -- Pythagoras"
wlhsi = wlhsi


def cheatsheet() -> str:
    return "wlhsi({}) -> Habitat suitability index"
