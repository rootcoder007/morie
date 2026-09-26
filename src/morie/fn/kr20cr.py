"""KR-20 reliability for binary items."""

__all__ = ["kuder_richardson_20"]


def kuder_richardson_20(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    KR-20 reliability for binary items

    Formula: KR-20 = k/(k-1) (1 - sum p_i q_i / sigma_T^2)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kuder-Richardson (1937)
    """
    raise NotImplementedError(
        "morie.fn.kr20cr.kuder_richardson_20 is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "kr20cr: KR-20 reliability for binary items"
