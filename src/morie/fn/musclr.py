"""MUSCLE multiple sequence alignment."""

__all__ = ["muscle_msa"]


def muscle_msa(sequences):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    MUSCLE multiple sequence alignment

    Formula: iterative refinement on guide tree

    Parameters
    ----------
    sequences : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Edgar (2004)
    """
    raise NotImplementedError(
        "morie.fn.musclr.muscle_msa is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "musclr: MUSCLE multiple sequence alignment"


# compact alias per ledger/NAMING.md
musclemsa = muscle_msa
