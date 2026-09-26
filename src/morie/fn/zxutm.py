"""UTM coordinate conversion"""


def utm_convert(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    UTM coordinate conversion

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxutm.utm_convert is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


utm_ = utm_convert


def cheatsheet() -> str:
    return "utm_convert({}) -> UTM coordinate conversion"


# compact alias per ledger/NAMING.md
utmconvert = utm_convert
