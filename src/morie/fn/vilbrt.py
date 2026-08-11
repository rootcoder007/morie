# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ViLBERT two-stream co-attention.

Same method as :mod:`morie.fn.hmvilb`: Lu, J., Batra, D., Parikh, D.
and Lee, S. (2019), "ViLBERT: Pretraining Task-Agnostic
Visiolinguistic Representations for Vision-and-Language Tasks",
NeurIPS 32, arXiv:1908.02265, Section 3.1 and Figure 2 -- two
transformer streams (image regions, text tokens) exchanging
information through co-attentional layers in which the QUERIES of one
modality attend over the KEYS and VALUES of the other (image queries
over text K/V and vice versa).

There is exactly one implementation: this module delegates to
:func:`morie.fn.hmvilb.geron_vilbert`.

Source: fetched-wave3/lu-etal-2019-vilbert-arxiv1908.02265.pdf
(Section 3.1, Figure 2).
"""

from .hmvilb import geron_vilbert as _impl

__all__ = ["vilbrt", "vilbert_two_stream"]


def vilbrt(image, text, d_model=8, seed=0):
    """ViLBERT co-attention (Lu et al. 2019, arXiv:1908.02265, Sec 3.1).

    Delegates to :func:`morie.fn.hmvilb.geron_vilbert`; see that
    function for parameters and payload.
    """
    return _impl(image, text, d_model=d_model, seed=seed)


vilbert_two_stream = vilbrt


def cheatsheet():
    return "vilbrt: ViLBERT two-stream co-attention (Lu et al. 2019, arXiv:1908.02265) -- alias of hmvilb.geron_vilbert"
