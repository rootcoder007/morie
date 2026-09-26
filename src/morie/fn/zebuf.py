"""Buffer-based exposure assessment"""


def buffer_exposure(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Buffer-based exposure assessment

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zebuf.buffer_exposure is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


buff = buffer_exposure


def cheatsheet() -> str:
    return "buffer_exposure({}) -> Buffer-based exposure assessment"


# compact alias per ledger/NAMING.md
bufferexposure = buffer_exposure
