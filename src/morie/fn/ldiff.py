# morie.fn -- function file (rootcoder007/morie)
"""l-diversity -- alias of the shipped implementation in dpld.

The generated stub described the l-diversity check of Machanavajjhala
et al. (2007).  ``dpld.l_diversity`` already implements distinct,
entropy and recursive (c, l)-diversity from that paper, so this module
aliases it.

References
----------
Machanavajjhala, A., Kifer, D., Gehrke, J., & Venkitasubramaniam, M.
    (2007). l-diversity: privacy beyond k-anonymity. *ACM Transactions
    on Knowledge Discovery from Data*, 1(1), article 3. Definitions in
    sections 4.1-4.2 (distinct, entropy, recursive (c, l)-diversity).
"""

from .dpld import l_diversity

__all__ = ["ldiff", "l_diversity_check"]

#: Primary name: alias of :func:`morie.fn.dpld.l_diversity`.
ldiff = l_diversity

#: Legacy stub name, kept for compatibility.
l_diversity_check = l_diversity


def cheatsheet():
    return "ldiff: l-diversity (alias of dpld.l_diversity)."
