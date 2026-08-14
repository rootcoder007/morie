# morie.fn -- function file (rootcoder007/morie)
r"""Named-entity recognition with BIO tagging.

Entities span several tokens, so a flat per-token classifier is not
enough: "New York City" is one entity, not three. The BIO scheme
encodes the boundary in the label itself -- ``B-TYPE`` opens an entity,
``I-TYPE`` continues the one already open, ``O`` is outside.

**Most label sequences are not valid, and a per-token argmax does not
know that.** ``I-PER`` cannot follow ``O``, and it cannot follow
``B-LOC`` either: an inside tag must continue an entity of its own
type. Decoding each position independently produces those sequences
routinely, and the usual response -- patching them up afterwards -- is
a silent re-scoring. The fix is to decode the whole sequence at once
under the constraint, which is what Viterbi over a transition matrix
does: forbidden transitions get :math:`-\infty` and the best *valid*
path is found exactly.

**The two decoders disagree, and the anchor makes them.** It builds
emissions whose greedy reading is invalid, checks the greedy path is
invalid and the Viterbi path is valid, and checks the Viterbi path
never scores higher than greedy on emissions alone -- because that is
the price of the constraint, and a Viterbi that beat greedy on
emissions would mean the constraint was not applied.

**Span extraction is the actual output.** Scoring is done over spans,
not tokens: a prediction counts only if its type *and* both boundaries
match, which is why ``B-``/``I-`` confusion is a real error rather than
a cosmetic one, and why token accuracy flatters a tagger that never
gets a boundary right.

References
----------
Ramshaw, L. A. & Marcus, M. P. (1995) "Text Chunking using
Transformation-Based Learning", *Proceedings of the Third Workshop on
Very Large Corpora*, 82-94, arXiv:cmp-lg/9505040. The origin of the
IOB/BIO tagging scheme.

Tjong Kim Sang, E. F. & De Meulder, F. (2003) "Introduction to the
CoNLL-2003 Shared Task: Language-Independent Named Entity
Recognition", *Proceedings of CoNLL-2003*, 142-147. The span-level
evaluation used here.

Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. & Dyer, C.
(2016) "Neural Architectures for Named Entity Recognition",
*Proceedings of NAACL-HLT 2016*, 260-270, doi:10.18653/v1/N16-1030,
arXiv:1603.01360. The BiLSTM-CRF whose constrained decoding this is.

Viterbi, A. J. (1967) "Error bounds for convolutional codes and an
asymptotically optimum decoding algorithm", *IEEE Transactions on
Information Theory* 13(2), 260-269, doi:10.1109/TIT.1967.1054010.

Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. (2019) "BERT:
Pre-training of Deep Bidirectional Transformers for Language
Understanding", *Proceedings of NAACL-HLT 2019*, 4171-4186,
doi:10.18653/v1/N19-1423. The encoder that usually supplies the
emissions.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["bio_labels", "valid_transitions", "viterbi_decode",
           "greedy_decode", "extract_spans", "span_f1", "is_valid_bio"]

_NEG = float("-inf")


def bio_labels(types):
    """The label set: O, then B-/I- for each entity type."""
    ts = list(types)
    if not ts:
        raise ValueError("benRea: no entity types given")
    if len(set(ts)) != len(ts):
        raise ValueError("benRea: duplicate entity types")
    out = ["O"]
    for t in ts:
        out.append("B-%s" % t)
        out.append("I-%s" % t)
    return out


def _parts(label):
    if label == "O":
        return "O", None
    return label[0], label[2:]


def valid_transitions(labels):
    r"""``T[a][b]`` is whether label b may follow label a.

    The one rule that matters: ``I-X`` may follow only ``B-X`` or
    ``I-X``. It may not follow ``O``, and it may not follow a tag of a
    different type.
    """
    n = len(labels)
    T = [[True] * n for _ in range(n)]
    for a in range(n):
        pa, ta = _parts(labels[a])
        for b in range(n):
            pb, tb = _parts(labels[b])
            if pb == "I":
                T[a][b] = (pa in ("B", "I")) and ta == tb
    return T


def start_allowed(labels):
    """An ``I-`` tag cannot open a sequence."""
    return [_parts(v)[0] != "I" for v in labels]


def is_valid_bio(path, labels=None):
    """Whether a label sequence obeys the scheme."""
    prev = "O"
    prev_t = None
    for lab in path:
        p, t = _parts(lab)
        if p == "I" and not (prev in ("B", "I") and prev_t == t):
            return False
        prev, prev_t = p, t
    return True


def greedy_decode(emissions, labels):
    """Per-token argmax -- which is free to produce invalid sequences,
    and does."""
    return [labels[max(range(len(labels)),
                       key=lambda j: emissions[t][j])]
            for t in range(len(emissions))]


def viterbi_decode(emissions, labels, transitions=None,
                   transition_scores=None):
    r"""The best VALID path, exactly.

    Forbidden transitions are :math:`-\infty`, so no valid-looking
    repair pass is needed afterwards -- the constraint is in the search.
    """
    L = len(emissions)
    n = len(labels)
    if L == 0:
        raise ValueError("benRea: empty emission sequence")
    if any(len(row) != n for row in emissions):
        raise ValueError("benRea: emissions must have one score per "
                         "label")
    T = valid_transitions(labels) if transitions is None else transitions
    S = ([[0.0] * n for _ in range(n)] if transition_scores is None
         else transition_scores)
    ok0 = start_allowed(labels)
    dp = [[_NEG] * n for _ in range(L)]
    bk = [[-1] * n for _ in range(L)]
    for j in range(n):
        if ok0[j]:
            dp[0][j] = emissions[0][j]
    for t in range(1, L):
        for j in range(n):
            best, arg = _NEG, -1
            for i in range(n):
                if not T[i][j] or dp[t - 1][i] == _NEG:
                    continue
                v = dp[t - 1][i] + S[i][j]
                if v > best:
                    best, arg = v, i
            if arg >= 0:
                dp[t][j] = best + emissions[t][j]
                bk[t][j] = arg
    end = max(range(n), key=lambda j: dp[L - 1][j])
    if dp[L - 1][end] == _NEG:
        raise ValueError("benRea: no valid path exists")
    path = [end]
    for t in range(L - 1, 0, -1):
        path.append(bk[t][path[-1]])
    path.reverse()
    return [labels[j] for j in path], dp[L - 1][end]


def extract_spans(path):
    """The (type, start, end) spans a label sequence encodes."""
    spans = []
    cur_t, cur_s = None, None
    for t, lab in enumerate(path):
        p, ty = _parts(lab)
        if p == "B":
            if cur_t is not None:
                spans.append((cur_t, cur_s, t - 1))
            cur_t, cur_s = ty, t
        elif p == "I":
            if cur_t != ty:
                # an orphan I- opens a span; the alternative is to drop
                # it, which silently hides the tagger's error
                if cur_t is not None:
                    spans.append((cur_t, cur_s, t - 1))
                cur_t, cur_s = ty, t
        else:
            if cur_t is not None:
                spans.append((cur_t, cur_s, t - 1))
            cur_t, cur_s = None, None
    if cur_t is not None:
        spans.append((cur_t, cur_s, len(path) - 1))
    return spans


def span_f1(pred, gold):
    """Span-level precision, recall and F1: a hit needs the type AND
    both boundaries."""
    p = set(extract_spans(pred))
    g = set(extract_spans(gold))
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "true_positives": tp, "n_pred": len(p), "n_gold": len(g)}


def ner_decode(emissions, types, decoder="viterbi",
               transition_scores=None, gold=None):
    """Decode a sentence and, if gold labels are given, score it."""
    if decoder not in ("viterbi", "greedy"):
        raise ValueError("benRea: decoder must be viterbi or greedy, "
                         "got %r" % (decoder,))
    labels = bio_labels(types)
    if decoder == "viterbi":
        path, score = viterbi_decode(emissions, labels,
                                     transition_scores=transition_scores)
    else:
        path = greedy_decode(emissions, labels)
        score = sum(emissions[t][labels.index(path[t])]
                    for t in range(len(path)))
    spans = extract_spans(path)
    payload = {"estimate": path, "path": path, "score": score,
               "spans": spans, "valid": is_valid_bio(path),
               "labels": labels, "decoder": decoder,
               "n_tokens": len(emissions), "n_spans": len(spans),
               "method": "BIO named-entity decoding, Ramshaw & Marcus "
                         "(1995) scheme, Viterbi (1967) constrained "
                         "decoding"}
    if gold is not None:
        payload.update(span_f1(path, list(gold)))
    return RichResult(payload=payload)


def cheatsheet():
    return ("benRea: BIO -- B- opens, I- continues, O outside. I-X may "
            "follow ONLY B-X or I-X, so a per-token argmax routinely "
            "emits invalid sequences. Viterbi with -inf on forbidden "
            "transitions finds the best VALID path exactly, and scores "
            "no higher on emissions than greedy -- that gap is the "
            "constraint. Score spans, not tokens.")


# compact alias per ledger/NAMING.md
nerdecode = ner_decode
__all__.append("ner_decode")

# public names resolved by fn/_lazy_map.json
named_entity = ner_decode
namedentity = ner_decode
