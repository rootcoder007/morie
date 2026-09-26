"""Travel time catchment"""


def travel_time_catch(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Travel time catchment

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zeitv.travel_time_catch is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


trav = travel_time_catch


def cheatsheet() -> str:
    return "travel_time_catch({}) -> Travel time catchment"
