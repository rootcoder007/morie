"""Indicator variogram"""


def indicator_vario(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Indicator variogram

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vginv.indicator_vario is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


indi = indicator_vario


def cheatsheet() -> str:
    return "indicator_vario({}) -> Indicator variogram"


# compact alias per ledger/NAMING.md
indicatorvario = indicator_vario
