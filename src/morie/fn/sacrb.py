# morie.fn -- function file (rootcoder007/morie)
r"""BLEU, and why the number alone is not comparable.

**The metric.** Modified n-gram precision clips each candidate n-gram
count at the maximum seen in any reference, so repeating a correct
word cannot inflate the score. Precision alone would reward
short output, so a multiplicative brevity penalty is applied. With
:math:`c` the total candidate length and :math:`r` the effective
reference length,

.. math:: BP = \begin{cases} 1 & c > r\\
          e^{(1 - r/c)} & c \le r \end{cases}, \qquad
          BLEU = BP \cdot \exp\Big(\sum_{n=1}^{N} w_n \log p_n\Big).

Two details in that definition matter and are easy to get wrong.

*The penalty is computed over the whole corpus, not per sentence.*
Averaging per-sentence penalties would punish length deviation on
short sentences far too harshly, so :math:`r` is the sum of best-match
lengths across the corpus and :math:`c` the total candidate length.
A sentence may be short if another is long.

*The best match length is the closest reference length, not the
shortest.* With references of 12, 15 and 17 words and a candidate of
12, the penalty is exactly 1.

**Why long candidates are not penalised twice.** Modified precision
already falls when the candidate is too long -- the extra n-grams are
unmatched. So :math:`BP = 1` for :math:`c > r` by design, not by
oversight.

**And now the part that makes the number usable.** BLEU is a function
of a *tokenisation*, and papers rarely report which one. The same
system, the same output and the same references produce materially
different BLEU depending on whether the text was tokenised with
Moses, split on whitespace, lowercased, or had punctuation detached --
and on whether the references were detokenised first. Scores from two
papers are therefore not comparable unless both name a scheme, which
they usually do not.

sacreBLEU's answer is not a better metric but a **protocol**: score
detokenised output against detokenised references, apply one named
tokenisation internally, and emit a version string recording every
choice. ``signature`` produces that string here. Two BLEU numbers are
comparable exactly when their signatures match, and the anchor
demonstrates the divergence rather than asserting it.

References
----------
Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. (2002) "BLEU: a
Method for Automatic Evaluation of Machine Translation", *Proceedings
of the 40th Annual Meeting of the Association for Computational
Linguistics (ACL 2002)*, 311-318, doi:10.3115/1073083.1073135.
Sec. 2.1 (modified n-gram precision with clipping), Sec. 2.2.2 (the
sentence brevity penalty, the "best match length" as the closest
reference length, and the argument for computing the penalty over the
corpus rather than per sentence), and Sec. 2.3 (the geometric mean
and the BP formula reproduced above).

Post, M. (2018) "A Call for Clarity in Reporting BLEU Scores",
*Proceedings of the Third Conference on Machine Translation (WMT18)*,
186-191, doi:10.18653/v1/W18-6319, arXiv:1804.08771. That BLEU depends on tokenisation and
preprocessing, that these are usually unreported, and that scores are
therefore not comparable across papers; the proposal to score
detokenised output with a single internal tokenisation and to publish
a version signature.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tokenize_13a", "tokenize_intl", "ngram_counts",
           "modified_precision", "brevity_penalty", "bleu",
           "signature"]

_EPS = 1e-12
_TOKENIZERS = ("13a", "intl", "none")
_PUNCT = ".,!?;:()\"'`-[]{}<>/\\|@#$%^&*+=~"


def tokenize_13a(text, lowercase=False):
    r"""The mteval-v13a scheme: detach punctuation, split on space."""
    s = str(text)
    if lowercase:
        s = s.lower()
    out = []
    for ch in s:
        out.append(" %s " % ch if ch in _PUNCT else ch)
    return "".join(out).split()


def tokenize_intl(text, lowercase=False):
    r"""A more aggressive scheme: every non-alphanumeric separates."""
    s = str(text)
    if lowercase:
        s = s.lower()
    out, cur = [], []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
            if not ch.isspace():
                out.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _tok(text, scheme, lowercase):
    if scheme == "13a":
        return tokenize_13a(text, lowercase)
    if scheme == "intl":
        return tokenize_intl(text, lowercase)
    if scheme == "none":
        s = str(text).lower() if lowercase else str(text)
        return s.split()
    raise ValueError("sacrb: tokenizer must be one of %s, got %r"
                     % (", ".join(_TOKENIZERS), scheme))


def ngram_counts(tokens, n):
    r"""Counts of every :math:`n`-gram in a token list."""
    if int(n) < 1:
        raise ValueError("sacrb: n must be at least 1")
    c = {}
    for i in range(len(tokens) - int(n) + 1):
        g = tuple(tokens[i:i + int(n)])
        c[g] = c.get(g, 0) + 1
    return c


def modified_precision(cand_tokens, refs_tokens, n):
    r"""Clipped precision: each n-gram counted at most as often as it
    appears in the single best reference.
    """
    cc = ngram_counts(cand_tokens, n)
    total = sum(cc.values())
    if total == 0:
        return {"numerator": 0, "denominator": 0, "precision": 0.0}
    best = {}
    for r in refs_tokens:
        rc = ngram_counts(r, n)
        for g, v in rc.items():
            if v > best.get(g, 0):
                best[g] = v
    clipped = sum(min(v, best.get(g, 0)) for g, v in cc.items())
    return {"numerator": clipped, "denominator": total,
            "precision": clipped / float(total)}


def brevity_penalty(c, r):
    r"""Sec. 2.2.2: 1 if :math:`c > r`, else :math:`e^{1 - r/c}`."""
    cv, rv = float(c), float(r)
    if cv <= 0.0:
        return 0.0
    return 1.0 if cv > rv else math.exp(1.0 - rv / cv)


def _best_match(clen, rlens):
    """The CLOSEST reference length, ties going to the shorter."""
    return min(rlens, key=lambda L: (abs(L - clen), L))


def bleu(candidates, references, max_n=4, weights=None,
         tokenizer="13a", lowercase=False):
    r"""Corpus BLEU, with the brevity penalty over the whole corpus.

    ``references`` is one list of reference strings per candidate.
    """
    C = [str(v) for v in candidates]
    R = [[str(x) for x in refs] for refs in references]
    if len(C) != len(R):
        raise ValueError("sacrb: %d candidates but %d reference sets"
                         % (len(C), len(R)))
    if not C:
        raise ValueError("sacrb: no candidates given")
    if any(not refs for refs in R):
        raise ValueError("sacrb: every candidate needs at least one "
                         "reference")
    N = int(max_n)
    if N < 1:
        raise ValueError("sacrb: max_n must be at least 1")
    w = ([1.0 / N] * N if weights is None
         else [float(v) for v in weights])
    if len(w) != N:
        raise ValueError("sacrb: %d weights for max_n = %d"
                         % (len(w), N))
    if abs(sum(w) - 1.0) > 1e-9:
        raise ValueError("sacrb: the weights must sum to 1, got %.6f"
                         % sum(w))
    num = [0] * N
    den = [0] * N
    c_total, r_total = 0, 0
    for i in range(len(C)):
        ct = _tok(C[i], tokenizer, lowercase)
        rt = [_tok(x, tokenizer, lowercase) for x in R[i]]
        c_total += len(ct)
        r_total += _best_match(len(ct), [len(x) for x in rt])
        for n in range(1, N + 1):
            mp = modified_precision(ct, rt, n)
            num[n - 1] += mp["numerator"]
            den[n - 1] += mp["denominator"]
    precisions = []
    for n in range(N):
        precisions.append(num[n] / den[n] if den[n] > 0 else 0.0)
    bp = brevity_penalty(c_total, r_total)
    if any(p <= 0.0 for p in precisions):
        score = 0.0
    else:
        score = bp * math.exp(sum(w[n] * math.log(precisions[n])
                                  for n in range(N)))
    return RichResult(payload={
        "estimate": score, "bleu": score, "score": 100.0 * score,
        "precisions": precisions, "bp": bp,
        "candidate_length": c_total, "reference_length": r_total,
        "ratio": c_total / float(max(r_total, 1)),
        "tokenizer": tokenizer, "lowercase": bool(lowercase),
        "max_n": N,
        "signature": signature(tokenizer, lowercase, N,
                               len(R[0])),
        "method": "corpus BLEU; Papineni et al. (2002) Sec. 2.3, "
                  "reported with a sacreBLEU-style signature "
                  "(Post 2018)",
    })


def signature(tokenizer="13a", lowercase=False, max_n=4, n_refs=1,
              version="morie-sacrb-1"):
    r"""The version string that makes two scores comparable.

    Post's point: BLEU is a function of the tokenisation, and without
    naming it the number is not comparable to anyone else's.
    """
    return ("nrefs:%d|case:%s|tok:%s|ngram:%d|version:%s"
            % (int(n_refs), "lc" if lowercase else "mixed",
               str(tokenizer), int(max_n), version))


def cheatsheet():
    return ("sacrb: BLEU = BP * exp(sum w_n log p_n), with clipped "
            "n-gram precision and BP = 1 if c > r else exp(1 - r/c). "
            "The penalty is computed over the WHOLE CORPUS, not per "
            "sentence, and r uses the CLOSEST reference length, not "
            "the shortest. Long candidates are not penalised twice -- "
            "modified precision already handles them. The number "
            "depends on TOKENISATION, so two BLEU scores are "
            "comparable only when their signatures match.")


# compact alias per ledger/NAMING.md
sacrebleu = bleu
