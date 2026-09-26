"""
UTM zone determination

Category: GeoProcss
"""


def utmzon(coords=None, n=50, source_crs="EPSG:4326", target_crs="EPSG:3857"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    UTM zone determination

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.utmzon.utmzon is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "utmzon"
alias = "utmzon"
quote = "It is not what happens to you, but how you react, that matters. -- Epictetus"
utmzon = utmzon


def cheatsheet() -> str:
    return "utmzon({}) -> UTM zone determination"
