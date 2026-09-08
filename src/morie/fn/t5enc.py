# morie.fn -- function file (rootcoder007/morie)
r"""T5: every task as text-to-text.

The unifying claim is a framing, not an architecture: **every** NLP
problem -- translation, classification, regression, summarisation --
is cast as feeding the model text and training it to produce text.
Classification emits the class *name*; regression emits a number as a
string, rounded to a fixed increment. One model, one loss
(teacher-forced maximum likelihood), one decoding procedure, and a
task prefix telling the model which job it is doing.

**What the framing buys, and what it costs.** It buys the ability to
mix every task in one training mixture and to transfer between them.
It costs the guarantee that the output is well formed: a
classification head cannot emit an invalid class, but a decoder can,
and the paper handles this by treating any output that is not a valid
label as wrong. ``parse_prediction`` reproduces that rule rather than
snapping to the nearest label, because snapping would hide the failure
mode the framing introduces.

**The pre-training objective is span corruption.** Rather than masking
single tokens, contiguous **spans** are dropped and replaced by a
single sentinel, and the target is the concatenation of the dropped
spans with their sentinels. That makes the target much shorter than
the input, which is a computational argument as much as a modelling
one -- and the paper's ablations pick 15% corruption with mean span
length 3.

**Relative position embeddings, shared across layers.** Position is
encoded as a learned scalar added to the attention logits, bucketed by
the *relative* offset between query and key, with buckets growing
logarithmically so distant positions share parameters. There is no
absolute position signal at all, which is what lets the model
generalise past the training length.

References
----------
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M.,
Zhou, Y., Li, W. & Liu, P. J. (2020) "Exploring the Limits of Transfer
Learning with a Unified Text-to-Text Transformer", *Journal of Machine
Learning Research* 21(140), 1-67, arXiv:1910.10683. The text-to-text
framework in which every task is fed text and produces text, with task
prefixes; the treatment of classification by emitting the label text
and of regression by emitting a rounded number as a string, with
outputs that do not match any label counted as wrong; the span
corruption pre-training objective with sentinels and the ablations
selecting 15% corruption and mean span length 3; and the simplified
relative position embeddings shared across layers, bucketed by
relative offset with logarithmically growing buckets.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
Gomez, A. N., Kaiser, L. & Polosukhin, I. (2017) "Attention Is All You
Need", *NIPS 2017*, 5998-6008, arXiv:1706.03762. The encoder-decoder
being modified.

Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019) "BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding", *NAACL-HLT 2019*, 4171-4186,
doi:10.18653/v1/N19-1423. The single-token masking objective span
corruption replaces.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["task_prefix", "span_corruption", "relative_bucket",
           "parse_prediction", "format_regression"]

_EPS = 1e-12


def task_prefix(task, text):
    r"""Prepend the prefix that tells the model which task this is."""
    t = str(task).strip()
    if not t:
        raise ValueError("t5enc: the task prefix cannot be empty -- "
                         "the model has no other signal of which job "
                         "it is doing")
    return "%s: %s" % (t, str(text))


def span_corruption(tokens, rate=0.15, mean_span=3.0, seed=0,
                    sentinel="<extra_id_%d>"):
    r"""Drop contiguous spans, replace each with one sentinel.

    The target is the dropped spans with their sentinels, so it is far
    shorter than the input -- which is a computational argument for
    spans over single tokens as much as a modelling one.
    """
    toks = [str(v) for v in tokens]
    n = len(toks)
    r = float(rate)
    if not 0.0 < r < 1.0:
        raise ValueError("t5enc: the corruption rate must lie in "
                         "(0,1), got %r" % (rate,))
    if float(mean_span) < 1.0:
        raise ValueError("t5enc: the mean span must be at least 1")
    n_corrupt = max(1, int(round(n * r)))
    n_spans = max(1, int(round(n_corrupt / float(mean_span))))
    rng = np.random.default_rng(seed)
    starts = sorted(set(int(float(rng.uniform()) * n) % n
                        for _ in range(n_spans * 3)))[:n_spans]
    spans, used = [], set()
    per = max(1, n_corrupt // max(len(starts), 1))
    for s in starts:
        e = min(n, s + per)
        if any(i in used for i in range(s, e)):
            continue
        spans.append((s, e))
        used |= set(range(s, e))
    spans.sort()
    src, tgt, idx, pos = [], [], 0, 0
    for (s, e) in spans:
        src.extend(toks[pos:s])
        src.append(sentinel % idx)
        tgt.append(sentinel % idx)
        tgt.extend(toks[s:e])
        idx += 1
        pos = e
    src.extend(toks[pos:])
    tgt.append(sentinel % idx)
    return {"input": src, "target": tgt, "n_spans": len(spans),
            "corrupted_tokens": len(used),
            "corruption_rate": len(used) / float(n),
            "target_shorter_by": len(src) - len(tgt),
            "note": "one sentinel per SPAN, so the target is much "
                    "shorter than the input"}


def relative_bucket(relative_position, bidirectional=True,
                    num_buckets=32, max_distance=128):
    r"""Bucket a relative offset, logarithmically for distant pairs.

    Nearby offsets get their own bucket; distant ones share, so the
    model extrapolates past the training length. There is no absolute
    position signal at all.
    """
    nb = int(num_buckets)
    if nb < 2:
        raise ValueError("t5enc: at least 2 buckets are needed")
    rp = int(relative_position)
    ret = 0
    if bidirectional:
        nb //= 2
        ret += nb if rp > 0 else 0
        rp = abs(rp)
    else:
        rp = -min(rp, 0)
    exact = nb // 2
    if rp < exact:
        return ret + rp
    v = exact + int(
        math.log(rp / float(exact))
        / math.log(float(max_distance) / exact) * (nb - exact))
    return ret + min(v, nb - 1)


def format_regression(value, increment=0.2, lo=1.0, hi=5.0):
    r"""Emit a number as a string, rounded to a fixed increment."""
    v = min(max(float(value), float(lo)), float(hi))
    inc = float(increment)
    if inc <= 0.0:
        raise ValueError("t5enc: the increment must be positive")
    return "%.1f" % (round(v / inc) * inc)


def parse_prediction(text, labels=None):
    r"""Parse the decoder's text; an invalid output is WRONG.

    Snapping to the nearest valid label would hide the failure mode
    the text-to-text framing introduces -- a decoder can emit
    something that is not a label at all.
    """
    s = str(text).strip()
    if labels is None:
        try:
            return {"value": float(s), "valid": True}
        except ValueError:
            return {"value": None, "valid": False,
                    "note": "not a number; counted as wrong"}
    ok = s in set(labels)
    return {"label": s if ok else None, "valid": ok,
            "note": "an output matching no label is counted as "
                    "WRONG, not snapped to the nearest one"}


def cheatsheet():
    return ("t5enc: EVERY task as text-to-text -- classification emits "
            "the label TEXT, regression emits a rounded number as a "
            "string, and a task prefix says which job it is. One "
            "model, one loss, one decoder, and tasks can be mixed. The "
            "cost: the decoder can emit something that is not a valid "
            "label, and that counts as WRONG rather than being snapped "
            "to the nearest one. Pre-training is SPAN corruption -- "
            "contiguous spans replaced by ONE sentinel each, so the "
            "target is far shorter (15%, mean span 3). Positions are "
            "RELATIVE, log-bucketed, shared across layers; there is no "
            "absolute position signal.")


# compact alias per ledger/NAMING.md
t5encoder = span_corruption

# public names resolved by fn/_lazy_map.json
t5 = span_corruption
