# morie.fn -- function file (rootcoder007/morie)
"""Exponential mechanism -- alias of the shipped implementation in dpexpm.

The generated stub described selection with probability proportional to
exp(epsilon u(D, r) / (2 Delta u)).  That mechanism already ships as
``dpexpm.dp_exponential_mechanism`` (McSherry-Talwar 2007; Dwork-Roth
Definition 3.4), so this module aliases it.

References
----------
McSherry, F., & Talwar, K. (2007). Mechanism design via differential
    privacy. *FOCS 2007*, 94-103.
Dwork, C., & Roth, A. (2014). The algorithmic foundations of
    differential privacy. *FnT-TCS*, 9(3-4), 211-487. Definition 3.4.
    Local source: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/dwork-roth-2014-algorithmic-foundations-differential-privacy.pdf
"""

from .dpexpm import dp_exponential_mechanism

__all__ = ["expmc", "exponential_mechanism"]

#: Primary name: alias of :func:`morie.fn.dpexpm.dp_exponential_mechanism`.
expmc = dp_exponential_mechanism

#: Legacy stub name, kept for compatibility.
exponential_mechanism = dp_exponential_mechanism


def cheatsheet():
    return "expmc: exponential mechanism (alias of dpexpm.dp_exponential_mechanism)."
