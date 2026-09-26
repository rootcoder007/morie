"""Flow-duration curve."""

__all__ = ["flow_duration"]


def flow_duration(Q):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Flow-duration curve

    Formula: sorted Q_t vs exceedance probability

    Parameters
    ----------
    Q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vogel-Fennessey (1995)
    """
    raise NotImplementedError(
        "morie.fn.floRate.flow_duration is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "floRate: Flow-duration curve"


# compact alias per ledger/NAMING.md
flowduration = flow_duration
