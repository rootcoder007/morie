# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""T5 span corruption: mask contiguous spans and predict them as one
target sequence."""

from ._richresult import RichResult

__all__ = ["kamath_t5_span_corruption"]


def _lcg(seed):
    """The deterministic generator used across this package's tests --
    no global RNG state, same masks on every machine."""
    s = int(seed) % 2 ** 32

    def nxt():
        nonlocal s
        s = (1664525 * s + 1013904223) % 2 ** 32
        return (s + 0.5) / 2 ** 32
    return nxt


def _segment(total, parts, rnd):
    """Split ``total`` items into ``parts`` positive-length pieces by
    choosing parts-1 dividers -- T5's random_spans_noise_mask, so the
    span lengths are random but every span is non-empty."""
    if parts > total:
        raise ValueError(
            f"cannot split {total} tokens into {parts} non-empty spans.")
    cuts = set()
    while len(cuts) < parts - 1:
        c = 1 + int(rnd() * (total - 1))
        c = min(c, total - 1)
        cuts.add(c)
    bounds = [0] + sorted(cuts) + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def kamath_t5_span_corruption(tokens, mean_span_len=3.0,
                              corruption_rate=0.15, seed=0,
                              sentinel="<extra_id_{}>"):
    """Input: the sequence with each masked span replaced by ONE
    sentinel; target: sentinel + span, repeated, then a final
    sentinel.

    The compression is the point -- a 3-token span costs one input
    token and the decoder has to produce all three, so the model
    cannot copy its way to a low loss. Deterministic given ``seed``.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, span corruption
    (T5, Raffel et al. 2020).

    Examples
    --------
    >>> toks = ["a", "b", "c", "d", "e", "f", "g", "h"]
    >>> out = kamath_t5_span_corruption(toks, mean_span_len=2.0,
    ...                                 corruption_rate=0.5, seed=7)
    >>> out["n_masked"], len(out["spans"])
    (4, 2)
    >>> len(sum(out["spans"], []))
    4
    >>> len(out["input"]) + 4 == len(toks) + 2
    True
    >>> out["input"].count("<extra_id_0>")
    1
    >>> out["target"][0]
    '<extra_id_0>'
    >>> out["target"][-1]
    '<extra_id_2>'
    """
    toks = list(tokens)
    L = len(toks)
    if L < 2:
        raise ValueError(
            f"need at least 2 tokens to corrupt a span; got {L}.")
    if not 0.0 < float(corruption_rate) < 1.0:
        raise ValueError(
            f"corruption_rate must lie in (0, 1); got {corruption_rate}. "
            "At 1 the model sees nothing to condition on.")
    if float(mean_span_len) < 1.0:
        raise ValueError(
            f"mean_span_len must be at least 1; got {mean_span_len}.")
    n_mask = int(round(L * float(corruption_rate)))
    n_mask = max(1, min(n_mask, L - 1))
    n_spans = int(round(n_mask / float(mean_span_len)))
    n_spans = max(1, min(n_spans, n_mask))
    n_keep = L - n_mask
    if n_keep < n_spans:
        raise ValueError(
            f"{n_spans} spans need at least {n_spans} unmasked tokens to "
            f"separate them, but only {n_keep} remain.")

    rnd = _lcg(seed)
    noise = _segment(n_mask, n_spans, rnd)
    keep = _segment(n_keep, n_spans, rnd)

    inp, tgt, spans = [], [], []
    pos = 0
    for i in range(n_spans):
        inp.extend(toks[pos:pos + keep[i]])
        pos += keep[i]
        span = toks[pos:pos + noise[i]]
        pos += noise[i]
        s = sentinel.format(i)
        inp.append(s)
        tgt.append(s)
        tgt.extend(span)
        spans.append(span)
    inp.extend(toks[pos:])
    tgt.append(sentinel.format(n_spans))
    return RichResult(payload={
        "input": inp, "target": tgt, "spans": spans,
        "n_masked": n_mask, "n_spans": n_spans,
        "span_lengths": noise,
        "compression": len(inp) / L,
        "estimate": n_mask, "n": L,
        "method": "T5 span corruption with sentinel tokens"})


def cheatsheet():
    return "kmspn: spans -> one sentinel each in the input, all of them in the target"
