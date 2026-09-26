"""SignalP signal peptide prediction."""

__all__ = ["signal_peptide"]


def signal_peptide(sequence):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    SignalP signal peptide prediction

    Formula: deep learning + cleavage site

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
    Almagro Armenteros et al (2019) SignalP-5
    """
    raise NotImplementedError(
        "morie.fn.sigpep.signal_peptide is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sigpep: SignalP signal peptide prediction"


# compact alias per ledger/NAMING.md
signalpeptide = signal_peptide
