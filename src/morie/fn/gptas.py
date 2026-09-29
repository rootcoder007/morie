# morie.fn -- function file (rootcoder007/morie)
"""Beam search decoding."""

from __future__ import annotations

from .hmbms import geron_beam_search

__all__ = ["gpt_assistant_decode"]


def gpt_assistant_decode(model, prompt, k=3, max_len=10, eos=None, length_penalty=0.0):
    """Beam search decoding.

    Keeps the ``k`` highest-scoring partial sequences, extends each by every
    token under ``model(prompt, prefix) -> log-probabilities`` and retains
    the best ``k`` until ``max_len`` or end-of-sequence (Graves 2012;
    Sutskever, Vinyals and Le 2014), with an optional length penalty. This
    is :func:`morie.fn.hmbms.geron_beam_search` with the prompt as source.

    References
    ----------
    Sutskever, I., Vinyals, O. and Le, Q. V. (2014). Sequence to sequence learning with neural networks.
    *NeurIPS*, 3104-3112.

    Graves, A. (2012). Sequence transduction with recurrent neural networks. arXiv:1211.3711.

    Examples
    --------
    >>> import math
    >>> def model(src, prefix):
    ...     return [math.log(0.6), math.log(0.4)] if len(prefix) % 2 == 0 else [math.log(0.3), math.log(0.7)]
    >>> gpt_assistant_decode(model, "hi", k=2, max_len=3)["sequence"]
    [0, 1, 0]
    """
    return geron_beam_search(model, prompt, beam_width=k, max_len=max_len, eos=eos, length_penalty=length_penalty)


def cheatsheet():
    return "gptas: beam search decoding (front-end to hmbms.geron_beam_search)"
