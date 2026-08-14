# morie.fn -- function file (rootcoder007/morie)
r"""Informer's ProbSparse attention -- duplicate ledger entry.

The wave-3 ledger carries this method twice, as ``infmer`` and as
``informer``, both citing Zhou et al. (2021) and both describing the
same sparse-attention forecaster. They are one paper and one method.

Rather than maintain two copies that could drift apart, this module
re-exports :mod:`informer`. Everything -- the query sparsity
measurement, the top-:math:`u` selection with :math:`u = c\ln L_Q`,
Lemma 1's max-mean approximation, and the complexity accounting --
lives there and is documented there.

References
----------
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H. & Zhang,
W. (2021) "Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting", *Proceedings of the AAAI Conference on
Artificial Intelligence* 35(12), 11106-11115, arXiv:2012.07436.

See Also
--------
:mod:`morie.fn.informer` -- the implementation.
"""

from .informer import (complexity, full_attention, kl_from_uniform,
                       probsparse_attention, select_queries,
                       sparsity_measure)

__all__ = ["sparsity_measure", "kl_from_uniform", "select_queries",
           "probsparse_attention", "full_attention", "complexity"]


def cheatsheet():
    from .informer import cheatsheet as _c
    return ("infmer: the same ledger method as `informer` -- one "
            "paper, one implementation, re-exported so the two "
            "entries cannot drift. " + _c())


# compact alias per ledger/NAMING.md
informerforecast = probsparse_attention

# public names resolved by fn/_lazy_map.json
informer = probsparse_attention
