"""
Water retention curve

Category: SoilSp
"""


def sowrc(data=None, depth=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Water retention curve

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.sowrc.sowrc is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "sowrc"
alias = "sowrc"
quote = "Number rules the universe. -- Pythagoras"
sowrc = sowrc


def cheatsheet() -> str:
    return "sowrc({}) -> Water retention curve"
