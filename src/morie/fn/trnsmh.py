"""TMHMM transmembrane topology."""

__all__ = ["transmembrane_topology"]


def transmembrane_topology(sequence):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    TMHMM transmembrane topology

    Formula: HMM with cytoplasmic / extracellular / TM states

    Parameters
    ----------
    sequence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Krogh et al (2001)
    """
    raise NotImplementedError(
        "morie.fn.trnsmh.transmembrane_topology is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "trnsmh: TMHMM transmembrane topology"
