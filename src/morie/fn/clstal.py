"""Clustal Omega progressive MSA."""

__all__ = ["clustalo"]


def clustalo(sequences):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Clustal Omega progressive MSA

    Formula: HHalign + guide tree + iterative refinement

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
    Sievers et al (2011)
    """
    raise NotImplementedError(
        "morie.fn.clstal.clustalo is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "clstal: Clustal Omega progressive MSA"
