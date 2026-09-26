"""
Available water storage

Category: SoilSp
"""


def soaws(data=None, depth=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Available water storage

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.soaws.soaws is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "soaws"
alias = "soaws"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
soaws = soaws


def cheatsheet() -> str:
    return "soaws({}) -> Available water storage"
