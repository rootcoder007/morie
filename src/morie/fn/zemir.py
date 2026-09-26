"""Migration flow model"""


def migration_flow(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Migration flow model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zemir.migration_flow is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


migr = migration_flow


def cheatsheet() -> str:
    return "migration_flow({}) -> Migration flow model"


# compact alias per ledger/NAMING.md
migrationflow = migration_flow
