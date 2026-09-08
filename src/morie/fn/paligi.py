# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ALiBi attention with per-head linear positional bias.

FABRICATED LEAD, RECORDED: the stub cited "Faisal & Anastasopoulos
(2022), parametric ALiBi with learnable per-head slopes" with the
formula a_ij = q_i k_j / sqrt(d) - sigmoid(s_h) |i - j|. No such paper
exists -- Fahim Faisal and Antonios Anastasopoulos have no 2022 (or
any) publication on ALiBi or attention slopes (checked their arXiv and
ACL Anthology records, 2026-08-09), and the sigmoid-slope formula is
unsourced. The real primary source for linear-bias attention is
Press, O., Smith, N. A. and Lewis, M. (2022), "Train Short, Test Long:
Attention with Linear Biases Enables Input Length Extrapolation",
ICLR 2022, arXiv:2108.12409 -- which moreover REJECTS trainable slopes
(Section 3: trainable slopes "did not yield strong extrapolation
results" and slowed training); the published method fixes head k of n
at m_k = 2^(-8k/n).

Implemented as published: this module delegates to
:mod:`morie.fn.atalib` (Press et al. 2022, page 4 modification
softmax(q_i K^T + m [-(i-1), ..., -1, 0]), geometric slope schedule).
Per-head parametrisation is available by passing explicit `slopes`,
which covers every published use without inventing a sigmoid.

Source: fetched-wave3/press-smith-lewis-2022-alibi-train-short-test-
long-arxiv2108.12409.pdf (page 4 and Section 3).
"""

from .atalib import alibi_position_bias as _impl
from .atalib import head_slopes

__all__ = ["paligi", "parametric_alibi", "head_slopes"]


def paligi(y=None, Q=None, K=None, V=None, slopes=None, causal=False):
    """ALiBi attention (Press, Smith and Lewis 2022, arXiv:2108.12409, p.4).

    Delegates to :func:`morie.fn.atalib.alibi_position_bias`; see that
    function for parameters and payload. Pass `slopes` for an explicit
    per-head slope; the default is the paper's geometric schedule.
    """
    return _impl(y=y, Q=Q, K=K, V=V, slopes=slopes, causal=causal)


parametric_alibi = paligi


def cheatsheet():
    return "paligi: ALiBi linear-bias attention (Press et al. 2022, arXiv:2108.12409) -- alias of atalib.alibi_position_bias; stub citation was fabricated"
