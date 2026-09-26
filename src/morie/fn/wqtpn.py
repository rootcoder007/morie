"""
Total nitrogen water

Category: WtrQual
"""


def wqtpn(data=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Total nitrogen water

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.wqtpn.wqtpn is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "wqtpn"
alias = "wqtpn"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
wqtpn = wqtpn


def cheatsheet() -> str:
    return "wqtpn({}) -> Total nitrogen water"
