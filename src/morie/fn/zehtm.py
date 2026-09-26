"""Hotspot detection map"""


def hotspot_map(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Hotspot detection map

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zehtm.hotspot_map is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


hots = hotspot_map


def cheatsheet() -> str:
    return "hotspot_map({}) -> Hotspot detection map"


# compact alias per ledger/NAMING.md
hotspotmap = hotspot_map
