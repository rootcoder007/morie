"""WGS84 to local tangent plane"""


def wgs84_to_local(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    WGS84 to local tangent plane

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxw84.wgs84_to_local is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


wgs8 = wgs84_to_local


def cheatsheet() -> str:
    return "wgs84_to_local({}) -> WGS84 to local tangent plane"


# compact alias per ledger/NAMING.md
wgs84tolocal = wgs84_to_local
