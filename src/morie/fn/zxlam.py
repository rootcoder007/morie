"""Lambert conformal conic projection"""


def lambert_proj(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Lambert conformal conic projection

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxlam.lambert_proj is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


lamb = lambert_proj


def cheatsheet() -> str:
    return "lambert_proj({}) -> Lambert conformal conic projection"


# compact alias per ledger/NAMING.md
lambertproj = lambert_proj
