"""Albers equal-area projection"""


def albers_proj(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Albers equal-area projection

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxalb.albers_proj is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


albe = albers_proj


def cheatsheet() -> str:
    return "albers_proj({}) -> Albers equal-area projection"


# compact alias per ledger/NAMING.md
albersproj = albers_proj
