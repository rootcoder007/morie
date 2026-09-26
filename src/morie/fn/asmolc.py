"""OLC long-read assembly."""

__all__ = ["olc_assembly"]


def olc_assembly(long_reads):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    OLC long-read assembly

    Formula: overlap-layout-consensus on long reads

    Parameters
    ----------
    long_reads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Myers (2005)
    """
    raise NotImplementedError(
        "morie.fn.asmolc.olc_assembly is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "asmolc: OLC long-read assembly"


# compact alias per ledger/NAMING.md
olcassembly = olc_assembly
