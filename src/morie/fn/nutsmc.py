# morie.fn -- function file (rootcoder007/morie)
"""No-U-Turn sampler (NUTS / HMC).

DUPLICATE: NUTS is already implemented in ``bnut`` (public name
``nuts_sampler``).  Per ledger/wave2/DUPMAP.tsv this module aliases it
rather than carrying a second sampler.
"""

from .bnut import nuts_sampler as _nuts

__all__ = ["nuts_sampler"]

nuts_sampler = _nuts


def cheatsheet():
    return "nutsmc: No-U-Turn sampler (alias of bnut.nuts_sampler)"


nutssampler = nuts_sampler
