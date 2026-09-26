# morie.fn -- function file (rootcoder007/morie)
"""First-order point pattern stats"""


def first_order_pp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    First-order point pattern stats

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.pt1st.first_order_pp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


firs = first_order_pp


def cheatsheet() -> str:
    return "first_order_pp({}) -> First-order point pattern stats"


# compact alias per ledger/NAMING.md
firstorderpp = first_order_pp
