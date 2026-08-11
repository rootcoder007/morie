# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BART denoising encoder-decoder (text-infilling corruption + scoring).

Same method as :mod:`morie.fn.hmbart`: Lewis, M., Liu, Y., Goyal, N.,
Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V. and
Zettlemoyer, L. (2020), "BART: Denoising Sequence-to-Sequence
Pre-training for Natural Language Generation, Translation, and
Comprehension", ACL 2020, arXiv:1910.13461, Section 2.2 -- the
text-infilling corruption samples span lengths from a Poisson
(lambda = 3) and replaces each contiguous span with a SINGLE <mask>
token, so the model must predict both content and span length; the
objective is reconstruction cross-entropy of the original text.

There is exactly one implementation: this module delegates to
:func:`morie.fn.hmbart.geron_bart`.

Source: fetched-wave3/lewis-etal-2020-bart-denoising-seq2seq-
arxiv1910.13461.pdf (Section 2.2, Figure 2).
"""

from .hmbart import geron_bart as _impl

__all__ = ["barte", "bart"]


def barte(src, tgt, mask_ratio=0.3, mean_span=3.0, permute=False, model=None, seed=0):
    """BART denoising pretraining step (Lewis et al. 2020, Sec 2.2).

    Delegates to :func:`morie.fn.hmbart.geron_bart`; see that function
    for parameters and payload.
    """
    return _impl(src, tgt, mask_ratio=mask_ratio, mean_span=mean_span,
                 permute=permute, model=model, seed=seed)


bart = barte


def cheatsheet():
    return "barte: BART text-infilling denoiser (Lewis et al. 2020, arXiv:1910.13461) -- alias of hmbart.geron_bart"
