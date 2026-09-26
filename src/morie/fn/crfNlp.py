"""Linear-chain CRF for sequence labeling."""

__all__ = ["crf_sequence"]


def crf_sequence(X, y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Linear-chain CRF for sequence labeling

    Formula: P(y|x) ∝ exp(sum θ_k f_k(y_t,y_{t-1},x))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lafferty-McCallum-Pereira (2001)
    """
    raise NotImplementedError(
        "morie.fn.crfNlp.crf_sequence is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "crfNlp: Linear-chain CRF for sequence labeling"


# compact alias per ledger/NAMING.md
crfsequence = crf_sequence
