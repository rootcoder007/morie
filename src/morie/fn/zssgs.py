"""Sequential Gaussian simulation"""


def seq_gauss_sim(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Sequential Gaussian simulation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zssgs.seq_gauss_sim is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


seq_ = seq_gauss_sim


def cheatsheet() -> str:
    return "seq_gauss_sim({}) -> Sequential Gaussian simulation"


# compact alias per ledger/NAMING.md
seqgausssim = seq_gauss_sim
