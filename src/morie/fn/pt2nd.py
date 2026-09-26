# morie.fn -- function file (rootcoder007/morie)
"""Second-order point pattern stats"""


def second_order_pp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Second-order point pattern stats

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.pt2nd.second_order_pp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


seco = second_order_pp


def cheatsheet() -> str:
    return "second_order_pp({}) -> Second-order point pattern stats"


# compact alias per ledger/NAMING.md
secondorderpp = second_order_pp
