# morie.fn -- function file (rootcoder007/morie)
"""Laplace mechanism -- alias of the shipped implementation in dpglap.

The generated stub for this module described the Laplace mechanism
M(D) = f(D) + Lap(sensitivity/epsilon).  That exact mechanism already
ships as ``dpglap.dp_laplace_mechanism`` (Dwork-Roth Definition 3.3),
so this module aliases it rather than adding a second implementation.

References
----------
Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating
    noise to sensitivity in private data analysis. *Theory of
    Cryptography (TCC 2006)*, LNCS 3876, 265-284.
Dwork, C., & Roth, A. (2014). The algorithmic foundations of
    differential privacy. *Foundations and Trends in Theoretical
    Computer Science*, 9(3-4), 211-487. Definition 3.3 and Theorem 3.6.
    Local source: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/dwork-roth-2014-algorithmic-foundations-differential-privacy.pdf
"""

from .dpglap import dp_laplace_mechanism

__all__ = ["laplc", "laplace_mechanism"]

#: Primary name: alias of :func:`morie.fn.dpglap.dp_laplace_mechanism`.
laplc = dp_laplace_mechanism

#: Legacy stub name, kept for compatibility.
laplace_mechanism = dp_laplace_mechanism


def cheatsheet():
    return "laplc: Laplace mechanism (alias of dpglap.dp_laplace_mechanism)."
