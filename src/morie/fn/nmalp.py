# morie.fn -- function file (rootcoder007/morie)
"""Alpha-NOMINATE posterior"""


def alpha_nom_post(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Alpha-NOMINATE posterior

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.nmalp.alpha_nom_post is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


alph = alpha_nom_post


def cheatsheet() -> str:
    return "alpha_nom_post({}) -> Alpha-NOMINATE posterior"


# compact alias per ledger/NAMING.md
alphanompost = alpha_nom_post
