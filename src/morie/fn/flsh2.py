# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FlashAttention -- IO-aware exact attention (tiled online softmax).

Same method as :mod:`morie.fn.hmfa`: Dao, T., Fu, D. Y., Ermon, S.,
Rudra, A. and Re, C. (2022), "FlashAttention: Fast and Memory-Efficient
Exact Attention with IO-Awareness", NeurIPS 35, arXiv:2205.14135,
Algorithm 1 -- block-tiled streaming of K/V with the running
(row max m, row sum l, accumulator O) triple rescaled by
exp(m_old - m_new) as the maximum moves, so the result equals
softmax(Q K^T / sqrt(d)) V exactly while only a block of scores is ever
materialised.

There is exactly one implementation: this module delegates to
:func:`morie.fn.hmfa.geron_flash_attention` (which itself checks the
tiled result against the direct softmax and reports max_abs_error).
Sibling copies of the same algorithm live in atfla/flsha/grflash;
hmfa is the delegate because it carries an R arm in both trees.

Source: fetched-wave3/dao-etal-2022-flashattention-arxiv2205.14135.pdf
(Algorithm 1, Section 3.1).
"""

from .hmfa import geron_flash_attention as _impl

__all__ = ["flsh2", "flash_attention"]


def flsh2(Q, K, V, block_size=2, causal=False):
    """FlashAttention (Dao et al. 2022, arXiv:2205.14135, Algorithm 1).

    Delegates to :func:`morie.fn.hmfa.geron_flash_attention`; see that
    function for parameters and payload.
    """
    return _impl(Q, K, V, block_size=block_size, causal=causal)


flash_attention = flsh2


def cheatsheet():
    return "flsh2: FlashAttention (Dao et al. 2022, arXiv:2205.14135) -- alias of hmfa.geron_flash_attention"
