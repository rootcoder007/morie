# morie.fn -- function file (rootcoder007/morie)
"""
D-infinity flow direction

Category: HydroSp
"""


def hydfl(flow=None, precip=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    D-infinity flow direction

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.hydfl.hydfl is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "hydfl"
alias = "hydfl"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
hydfl = hydfl


def cheatsheet() -> str:
    return "hydfl({}) -> D-infinity flow direction"
