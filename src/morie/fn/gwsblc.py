"""GWAS results across blocks."""

__all__ = ["gwas_block_combine"]


def gwas_block_combine(block_results):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    GWAS results across blocks

    Formula: meta-analysis of per-block stats

    Parameters
    ----------
    block_results : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Bakker et al (2008)
    """
    raise NotImplementedError(
        "morie.fn.gwsblc.gwas_block_combine is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "gwsblc: GWAS results across blocks"
