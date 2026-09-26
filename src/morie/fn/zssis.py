"""Sequential indicator simulation"""


def seq_ind_sim(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Sequential indicator simulation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zssis.seq_ind_sim is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


seq_ = seq_ind_sim


def cheatsheet() -> str:
    return "seq_ind_sim({}) -> Sequential indicator simulation"


# compact alias per ledger/NAMING.md
seqindsim = seq_ind_sim
