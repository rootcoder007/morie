"""Spatial contact tracing"""


def contact_trace_sp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial contact tracing

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zectr.contact_trace_sp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cont = contact_trace_sp


def cheatsheet() -> str:
    return "contact_trace_sp({}) -> Spatial contact tracing"


# compact alias per ledger/NAMING.md
contacttracesp = contact_trace_sp
