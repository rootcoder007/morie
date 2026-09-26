"""Inverse logit back to proportion."""

__all__ = ["ma_logit_inverse"]


def ma_logit_inverse(z):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Inverse logit back to proportion

    Formula: p = exp(z)/(1+exp(z))

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Nyaga et al. (2014)
    """
    raise NotImplementedError(
        "morie.fn.matransi.ma_logit_inverse is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "matransi: Inverse logit back to proportion"


# compact alias per ledger/NAMING.md
malogitinverse = ma_logit_inverse
