# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BART: denoising autoencoder pretraining for seq2seq."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bart"]

_MASK = "<mask>"


def _lcg_stream(count, seed):
    s = int(seed) % 2**32
    out = np.empty(int(count))
    for i in range(int(count)):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s + 0.5) / 2**32
    return out


def _bigram_lm(tgt, vocab):
    """Add-one smoothed bigram LM over `vocab`, estimated on the target."""
    V = len(vocab)
    index = {tok: i for i, tok in enumerate(vocab)}
    counts = np.ones((V + 1, V))  # row V is the start-of-sequence context
    for i, tok in enumerate(tgt):
        prev = V if i == 0 else index[tgt[i - 1]]
        counts[prev, index[tok]] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return index, probs


def geron_bart(src, tgt, mask_ratio=0.3, mean_span=3.0, permute=False, model=None, seed=0):
    """
    BART: denoising autoencoder pretraining for seq2seq.

    Formula: corrupt text -> encoder -> reconstruct via decoder

    Implements the text-infilling corruption exactly as BART defines it --
    contiguous spans are replaced by a *single* ``<mask>`` token, so the
    decoder must recover both the content and the length -- and then scores
    the reconstruction of `tgt`. With no `model` supplied the scorer is an
    add-one-smoothed bigram language model estimated on the target, which
    gives a genuine cross-entropy in nats per token; pass your own
    ``model(corrupted_src, tgt) -> per-token log-probs`` to score a real
    seq2seq decoder, and its length and finiteness are enforced.

    Parameters
    ----------
    src : sequence
        Source tokens (any hashable tokens).
    tgt : sequence
        Target tokens to reconstruct.
    mask_ratio : float
        Fraction of source tokens to cover with spans, in (0, 1).
    mean_span : float
        Mean span length (>= 1); span lengths are drawn from a geometric
        distribution with that mean.
    permute : bool
        Also apply the sentence-permutation noise (rotate the token list).
    model : callable, optional
        Custom scorer as described above.
    seed : int
        LCG seed for the span sampling.

    Returns
    -------
    result : RichResult
        Keys: corrupted, spans, n_masked, loss, perplexity, estimate, n, method.

    Examples
    --------
    >>> src = ["the", "cat", "sat", "on", "the", "mat", "today", "ok"]
    >>> tgt = ["the", "cat", "sat", "on", "the", "mat"]
    >>> r = geron_bart(src, tgt, mask_ratio=0.25, seed=2)
    >>> r["n_masked"]
    2
    >>> r["corrupted"].count("<mask>") >= 1
    True

    Each span collapses to one mask token, so the corrupted sequence is
    strictly shorter whenever a span is longer than one token:

    >>> bool(len(r["corrupted"]) <= len(src))
    True

    The default scorer is a real bigram model, so a target with predictable
    structure costs far fewer nats per token than one with no repeats:

    >>> rep = geron_bart(src, ["a", "b", "a", "b", "a", "b"], seed=2)["loss"]
    >>> dis = geron_bart(src, ["a", "b", "c", "d", "e", "f"], seed=2)["loss"]
    >>> bool(rep < dis)
    True
    >>> bool(dis < np.log(6))
    True

    References
    ----------
    Géron Ch 15
    """
    s_toks = list(src)
    t_toks = list(tgt)
    if not s_toks:
        raise ValueError("geron_bart: src is empty")
    if not t_toks:
        raise ValueError("geron_bart: tgt is empty")
    r = float(mask_ratio)
    if not (0.0 < r < 1.0):
        raise ValueError(f"geron_bart: mask_ratio must lie in (0, 1), got {r}")
    ms = float(mean_span)
    if ms < 1.0:
        raise ValueError(f"geron_bart: mean_span must be >= 1, got {ms}")

    budget = max(1, int(round(r * len(s_toks))))
    u = _lcg_stream(4 * len(s_toks) + 8, seed + 17)
    covered = np.zeros(len(s_toks), dtype=bool)
    spans = []
    ui = 0
    guard = 0
    while covered.sum() < budget and guard < 10 * len(s_toks):
        guard += 1
        start = int(u[ui % len(u)] * len(s_toks))
        ui += 1
        # Geometric span length with the requested mean.
        p = 1.0 / ms
        length = 1 if p >= 1.0 else int(np.floor(np.log(max(u[ui % len(u)], 1e-12)) / np.log(1.0 - p))) + 1
        ui += 1
        length = max(1, min(length, budget - int(covered.sum()), len(s_toks) - start))
        if covered[start : start + length].any():
            continue
        covered[start : start + length] = True
        spans.append((start, length))

    spans.sort()
    corrupted = []
    i = 0
    span_at = {st: ln for st, ln in spans}
    while i < len(s_toks):
        if i in span_at:
            corrupted.append(_MASK)
            i += span_at[i]
        else:
            corrupted.append(s_toks[i])
            i += 1
    if permute:
        k = int(u[-1] * len(corrupted))
        corrupted = corrupted[k:] + corrupted[:k]

    if model is None:
        vocab = sorted({str(t) for t in t_toks})
        index, probs = _bigram_lm([str(t) for t in t_toks], vocab)
        V = len(vocab)
        logps = []
        for i, tok in enumerate([str(t) for t in t_toks]):
            prev = V if i == 0 else index[str(t_toks[i - 1])]
            logps.append(float(np.log(probs[prev, index[tok]])))
        logps = np.asarray(logps)
    else:
        if not callable(model):
            raise ValueError("geron_bart: model must be callable")
        logps = np.asarray(model(corrupted, t_toks), dtype=float).ravel()
        if logps.size != len(t_toks):
            raise ValueError(
                f"geron_bart: model returned {logps.size} token log-probs for a target of length {len(t_toks)}"
            )
        if not np.all(np.isfinite(logps)):
            raise ValueError("geron_bart: model returned non-finite log-probabilities")
        if np.any(logps > 0):
            raise ValueError("geron_bart: model returned positive log-probabilities")

    loss = float(-np.mean(logps))

    return RichResult(
        title="BART denoising pretraining",
        summary_lines=[
            ("Masked tokens", int(covered.sum())),
            ("Spans", len(spans)),
            ("Reconstruction loss (nats/token)", loss),
        ],
        payload={
            "corrupted": corrupted,
            "spans": spans,
            "n_masked": int(covered.sum()),
            "n_spans": len(spans),
            "loss": loss,
            "perplexity": float(np.exp(loss)),
            "token_logprobs": logps,
            "estimate": loss,
            "n": int(len(t_toks)),
            "method": "BART text-infilling corruption scored by seq2seq reconstruction cross-entropy",
        },
    )


def cheatsheet():
    return "hmbart: BART: denoising autoencoder pretraining for seq2seq"


# compact alias per ledger/NAMING.md
geronbart = geron_bart
