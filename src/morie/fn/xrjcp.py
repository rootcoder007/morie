"""Join count permutation test"""


def join_count_perm(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Join count permutation test

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrjcp.join_count_perm is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


join = join_count_perm


def cheatsheet() -> str:
    return "join_count_perm({}) -> Join count permutation test"


# compact alias per ledger/NAMING.md
joincountperm = join_count_perm
