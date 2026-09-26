"""tiktoken-style efficient BPE."""

__all__ = ["tiktoken_bpe"]


def tiktoken_bpe(corpus):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    tiktoken-style efficient BPE

    Formula: BPE with rust regex pre-tokenizer

    Parameters
    ----------
    corpus : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    OpenAI tiktoken (2022)
    """
    raise NotImplementedError(
        "morie.fn.tikto.tiktoken_bpe is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "tikto: tiktoken-style efficient BPE"


# compact alias per ledger/NAMING.md
tiktokenbpe = tiktoken_bpe
