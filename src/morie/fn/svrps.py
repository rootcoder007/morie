"""Ranked probability score spatial"""


def rank_prob_score(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ranked probability score spatial

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svrps.rank_prob_score is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


rank = rank_prob_score


def cheatsheet() -> str:
    return "rank_prob_score({}) -> Ranked probability score spatial"


# compact alias per ledger/NAMING.md
rankprobscore = rank_prob_score
