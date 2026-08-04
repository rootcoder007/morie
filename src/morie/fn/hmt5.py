# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""T5: text-to-text transfer transformer (encoder-decoder)."""

from ._richresult import RichResult

__all__ = ["geron_t5", "span_corrupt", "restore"]

SENTINEL = "<extra_id_{}>"


def _tokens(x, name):
    if isinstance(x, str):
        out = x.split()
    else:
        out = [str(t) for t in x]
    if not out:
        raise ValueError(f"geron_t5: {name} is empty")
    return out


def span_corrupt(tokens, noise_density=0.15, mean_span=3, seed=0):
    """Mask contiguous spans and emit the T5 encoder input / decoder target.

    Spans are chosen from a deterministic LCG stream. The encoder input
    keeps one sentinel per dropped span; the decoder target is the dropped
    spans, each introduced by the matching sentinel and terminated by a
    final one. Together they contain every original token exactly once,
    which is what makes the objective lossless.
    """
    n = len(tokens)
    dens = float(noise_density)
    if not (0.0 < dens < 1.0):
        raise ValueError(f"span_corrupt: noise_density must lie in (0, 1), got {dens}")
    span = int(mean_span)
    if span < 1:
        raise ValueError(f"span_corrupt: mean_span must be >= 1, got {span}")
    n_noise = max(1, int(round(n * dens)))
    n_spans = max(1, int(round(n_noise / span)))
    if n_noise >= n:
        raise ValueError(
            f"span_corrupt: noise_density {dens} would mask all {n} tokens; nothing would be left to condition on"
        )

    s = int(seed) % 2**32
    chosen = []
    used = set()
    guard = 0
    while len(chosen) < n_spans and guard < 100 * n_spans:
        guard += 1
        s = (1664525 * s + 1013904223) % 2**32
        start = int(((s + 0.5) / 2**32) * n)
        length = max(1, min(span, n - start, n_noise - sum(l for _, l in chosen)))
        if length < 1 or any(i in used for i in range(start, start + length)):
            continue
        if start + length > n or (start == 0 and length == n):
            continue
        chosen.append((start, length))
        used.update(range(start, start + length))
        if sum(l for _, l in chosen) >= n_noise:
            break
    if not chosen:
        raise ValueError("span_corrupt: could not place any span; try a larger sequence or smaller mean_span")
    chosen.sort()

    inputs, target = [], []
    i = k = 0
    for start, length in chosen:
        inputs.extend(tokens[i:start])
        inputs.append(SENTINEL.format(k))
        target.append(SENTINEL.format(k))
        target.extend(tokens[start : start + length])
        i = start + length
        k += 1
    inputs.extend(tokens[i:])
    target.append(SENTINEL.format(k))
    return inputs, target, chosen


def restore(inputs, target):
    """Rebuild the original sequence from a corrupted input and its target."""
    spans = {}
    cur = None
    for t in target:
        if t.startswith("<extra_id_"):
            cur = t
            spans[cur] = []
        elif cur is not None:
            spans[cur].append(t)
    out = []
    for t in inputs:
        if t.startswith("<extra_id_"):
            out.extend(spans.get(t, []))
        else:
            out.append(t)
    return out


def geron_t5(src, tgt=None, noise_density=0.15, mean_span=3, seed=0, prefix="translate:"):
    """
    T5: text-to-text transfer transformer (encoder-decoder).

    Formula: every task cast as text-to-text; span corruption pretraining

    Both halves of T5's design are implemented:

    * **Text-to-text framing.** Any supervised pair becomes
      ``(prefix + source) -> target`` strings, so classification,
      translation and summarisation share one loss and one decoder. The
      framed pair is returned when `tgt` is supplied.
    * **Span corruption pretraining.** Contiguous spans are dropped from
      the input and replaced by numbered sentinels; the target is the
      dropped spans, each introduced by its sentinel (see
      :func:`span_corrupt`). This is denser than BERT's single-token
      masking -- one sentinel stands for several tokens, so the decoder
      predicts fewer, longer pieces.

    The objective is lossless, and that is verified rather than asserted:
    :func:`restore` rebuilds the original sequence from the corrupted
    input plus the target, and the round trip is checked.

    Parameters
    ----------
    src : str or sequence
        Source sequence (whitespace-split when a string).
    tgt : str or sequence, optional
        Target sequence for the text-to-text framing.
    noise_density : float, default 0.15
        Fraction of tokens to mask, in (0, 1).
    mean_span : int, default 3
        Mean span length (>= 1).
    seed : int, default 0
        LCG seed for span placement.
    prefix : str, default "translate:"
        Task prefix used in the text-to-text framing.

    Returns
    -------
    result : RichResult
        Keys: encoder_input, decoder_target, spans, restored, lossless,
        n_masked, sentinels, text_to_text, estimate, n, method.

    Examples
    --------
    >>> r = geron_t5("the quick brown fox jumps over the lazy dog", noise_density=0.3, mean_span=2)
    >>> bool(r["lossless"])
    True
    >>> " ".join(r["restored"]) == "the quick brown fox jumps over the lazy dog"
    True
    >>> int(r["sentinels"]) >= 1
    True
    >>> sum(1 for t in r["encoder_input"] if t.startswith("<extra_id_")) == r["sentinels"]
    True

    The text-to-text framing turns a supervised pair into two strings:

    >>> r2 = geron_t5("hello there my friend", "bonjour mon ami", prefix="translate English to French:")
    >>> r2["text_to_text"]
    ('translate English to French: hello there my friend', 'bonjour mon ami')

    References
    ----------
    Géron Ch 15
    """
    toks = _tokens(src, "src")
    if len(toks) < 2:
        raise ValueError("geron_t5: span corruption needs at least 2 source tokens")
    enc, dec, spans = span_corrupt(toks, noise_density, mean_span, seed)
    rebuilt = restore(enc, dec)
    lossless = rebuilt == toks

    t2t = None
    if tgt is not None:
        tgt_toks = _tokens(tgt, "tgt")
        t2t = (f"{prefix} {' '.join(toks)}".strip(), " ".join(tgt_toks))

    n_masked = int(sum(l for _, l in spans))

    return RichResult(
        title="T5 span corruption",
        summary_lines=[
            ("Source tokens", len(toks)),
            ("Masked tokens", n_masked),
            ("Spans", len(spans)),
            ("Lossless round trip", lossless),
        ],
        interpretation=(
            "One sentinel replaces a whole span, so the decoder emits far fewer tokens than BERT's "
            "per-token masking would -- cheaper pretraining for the same corrupted fraction."
        ),
        payload={
            "encoder_input": enc,
            "decoder_target": dec,
            "spans": spans,
            "restored": rebuilt,
            "lossless": bool(lossless),
            "n_masked": n_masked,
            "sentinels": len(spans),
            "text_to_text": t2t,
            "estimate": float(n_masked / len(toks)),
            "n": int(len(toks)),
            "method": "T5 span corruption with sentinel tokens, verified by exact reconstruction; text-to-text framing",
        },
    )


def cheatsheet():
    return "hmt5: T5: text-to-text transfer transformer (encoder-decoder)"


# compact alias per ledger/NAMING.md
geront5 = geron_t5


# compact alias per ledger/NAMING.md
spancorrupt = span_corrupt
