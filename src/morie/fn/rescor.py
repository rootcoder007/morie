"""Consensus rescoring across multiple docking functions."""

__all__ = ["rescore_consensus"]


def rescore_consensus(scores):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Consensus rescoring across multiple docking functions

    Formula: rank-sum / Borda count over Vina + Glide + ChemScore

    Parameters
    ----------
    scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Charifson et al (1999)
    """
    raise NotImplementedError(
        "morie.fn.rescor.rescore_consensus is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "rescor: Consensus rescoring across multiple docking functions"
