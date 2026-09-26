"""Mercator projection"""


def mercator_proj(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Mercator projection

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxmrc.mercator_proj is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


merc = mercator_proj


def cheatsheet() -> str:
    return "mercator_proj({}) -> Mercator projection"


# compact alias per ledger/NAMING.md
mercatorproj = mercator_proj
