"""Multi-candidate Hotelling model"""


def hotelling_multi(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Multi-candidate Hotelling model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svhtm.hotelling_multi is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


hote = hotelling_multi


def cheatsheet() -> str:
    return "hotelling_multi({}) -> Multi-candidate Hotelling model"


# compact alias per ledger/NAMING.md
hotellingmulti = hotelling_multi
