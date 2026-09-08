# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPT-1: decoder-only transformer pretrained on next-token prediction."""

from . import _array_core as np

from ._richresult import RichResult
from .hmdctr import geron_decoder_only

__all__ = ["geron_gpt1"]

# GPT-1 (Radford et al. 2018): 12 layers, 12 heads, d_model 768, BPE 40478,
# context 512, FFN width 3072.
_CONFIG = {"n_layers": 12, "n_heads": 12, "d_model": 768, "vocab_size": 40478, "max_len": 512, "d_ff": 3072}


def geron_gpt1(X, n_layers=None, n_heads=None, logits=None, targets=None, **config):
    """
    GPT-1: decoder-only transformer pretrained on next-token prediction.

    Formula: L = -sum_t log P(x_t | x_{<t})

    The architecture is DELEGATED to
    :func:`morie.fn.hmdctr.geron_decoder_only` with the GPT-1
    configuration (12 layers, 12 heads, ``d_model`` 768, context 512,
    BPE vocabulary 40478); overriding ``n_layers`` or ``n_heads`` scales
    it while keeping everything else consistent.

    What this module adds is the objective in the formula line, computed
    for real when ``logits`` are supplied: the causal language-modelling
    loss is the mean over positions of ``-log softmax(logits_t)[x_t]``,
    with the shift built in -- position ``t`` predicts token ``t+1``, so
    the last position has no target and the first token is never
    predicted. That off-by-one is where LM losses are usually wrong, so
    ``n_predicted`` is reported.

    Parameters
    ----------
    X : array-like
        Token ids.
    n_layers, n_heads : int, optional
        Override the GPT-1 defaults.
    logits : array-like, shape (T, V), optional
        Model logits for the sequence; enables the loss computation.
    targets : array-like, optional
        Next-token targets; default the shifted ``X``.
    **config
        Any other ``geron_decoder_only`` argument.

    Returns
    -------
    result : RichResult
        Keys: total_params, config, loss, perplexity, token_losses,
        n_predicted, mask, estimate, n, method.

    Examples
    --------
    The published configuration comes to 116.5M parameters with tied
    embeddings:

    >>> r = geron_gpt1(list(range(8)))
    >>> r["config"]["n_layers"], r["config"]["d_model"], r["config"]["max_len"]
    (12, 768, 512)
    >>> r["total_params"]
    116536320

    A uniform model over two tokens costs ``log 2`` per predicted
    position, and there are ``T - 1`` of them:

    >>> import math
    >>> r2 = geron_gpt1([0, 1, 0], logits=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ...                 n_layers=1, n_heads=1, d_model=2, vocab_size=2, max_len=4)
    >>> round(r2["loss"], 9) == round(math.log(2), 9)
    True
    >>> r2["n_predicted"]
    2
    >>> round(r2["perplexity"], 6)
    2.0

    A confident correct model costs almost nothing:

    >>> r3 = geron_gpt1([0, 1], logits=[[0.0, 10.0], [0.0, 0.0]],
    ...                 n_layers=1, n_heads=1, d_model=2, vocab_size=2, max_len=4)
    >>> f"{r3['loss']:.6f}"
    '0.000045'

    References
    ----------
    Géron Ch 15
    """
    cfg = dict(_CONFIG)
    cfg.update({k: v for k, v in config.items() if v is not None})
    if n_layers is not None:
        cfg["n_layers"] = int(n_layers)
    if n_heads is not None:
        cfg["n_heads"] = int(n_heads)
    arch = geron_decoder_only(X, **cfg)

    A = np.asarray(X).ravel()
    loss = ppl = None
    tok = None
    n_pred = int(max(A.size - 1, 0))
    if logits is not None:
        Z = np.atleast_2d(np.asarray(logits, dtype=float))
        if Z.shape[0] != A.size:
            raise ValueError(f"geron_gpt1: logits has {Z.shape[0]} rows but the sequence has {A.size} tokens")
        if not np.all(np.isfinite(Z)):
            raise ValueError("geron_gpt1: logits contains non-finite values")
        if n_pred < 1:
            raise ValueError("geron_gpt1: a sequence of one token has nothing to predict")
        y = A[1:].astype(int) if targets is None else np.asarray(targets).ravel().astype(int)
        if y.size != n_pred:
            raise ValueError(f"geron_gpt1: expected {n_pred} targets (one per predicted position), got {y.size}")
        if y.min() < 0 or y.max() >= Z.shape[1]:
            raise ValueError(f"geron_gpt1: a target lies outside the logit vocabulary of {Z.shape[1]}")
        Zs = Z[:-1]
        shift = Zs - Zs.max(axis=1, keepdims=True)
        logZ = np.log(np.exp(shift).sum(axis=1))
        tok = (logZ - shift[np.arange(n_pred), y]).tolist()
        loss = float(np.mean(tok))
        ppl = float(np.exp(loss))

    return RichResult(
        title="GPT-1",
        summary_lines=[("Parameters", int(arch["total_params"])), ("Layers", cfg["n_layers"]), ("Loss", loss)],
        interpretation="Position t predicts token t+1, so a T-token sequence supplies T-1 training signals.",
        payload={
            "total_params": int(arch["total_params"]),
            "config": cfg,
            "block_params": int(arch["block_params"]),
            "embedding_params": int(arch["embedding_params"]),
            "d_head": int(arch["d_head"]),
            "mask": arch["mask"],
            "loss": loss,
            "perplexity": ppl,
            "token_losses": tok,
            "n_predicted": n_pred,
            "estimate": float(arch["total_params"]) if loss is None else loss,
            "n": int(A.size),
            "method": "GPT-1 architecture delegated to hmdctr, plus the causal LM objective",
        },
    )


def cheatsheet():
    return "hmgpt1: GPT-1: decoder-only transformer pretrained on next-token prediction"


# compact alias per ledger/NAMING.md
gerongpt1 = geron_gpt1
