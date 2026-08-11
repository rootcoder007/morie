# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPT-2 decoder-only forward pass (released size configurations).

Same method as :mod:`morie.fn.hmgpt2`: Radford, A., Wu, J., Child, R.,
Luan, D., Amodei, D. and Sutskever, I. (2019), "Language Models are
Unsupervised Multitask Learners", OpenAI technical report -- the
decoder-only transformer of Radford et al. (2018, GPT) scaled to the
four released sizes (117M/345M/762M/1542M; BPE vocabulary 50257,
context 1024). The stub cited "Radford et al (2018) GPT" for a
"GPT-style decoder forward pass"; the forward pass itself is
DELEGATED by the target to morie.fn.hmdctr.geron_decoder_only
(masked self-attention blocks + LM head), and hmgpt2 adds the
released configurations and parameter-scaling arithmetic.

There is exactly one implementation: this module delegates to
:func:`morie.fn.hmgpt2.geron_gpt2`.

Source: fetched-wave3/radford-etal-2019-gpt2-unsupervised-multitask-
learners.pdf (Section 2.3, Table 2 for the sizes).
"""

from .hmgpt2 import geron_gpt2 as _impl

__all__ = ["gpt2", "gpt_decoder"]


def gpt2(X, n_layers=None, n_heads=None, size="small", **config):
    """GPT-2 decoder-only LM (Radford et al. 2019, Sec 2.3, Table 2).

    Delegates to :func:`morie.fn.hmgpt2.geron_gpt2`; see that function
    for parameters and payload.
    """
    return _impl(X, n_layers=n_layers, n_heads=n_heads, size=size, **config)


gpt_decoder = gpt2


def cheatsheet():
    return "gpt2: GPT-2 decoder-only LM (Radford et al. 2019) -- alias of hmgpt2.geron_gpt2"
