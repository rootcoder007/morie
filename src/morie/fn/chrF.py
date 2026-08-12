r"""chrF: character n-gram F-score for machine translation.

Popović, M. (2015) "chrF: character n-gram F-score for automatic MT
evaluation", *Proc. 10th Workshop on Statistical Machine Translation
(WMT15)*, 392-395.

Character n-grams rather than word n-grams, which is what makes the
metric work on morphologically rich languages where a word-level metric
scores a correct inflection as a total miss.

.. math:: \mathrm{chrP} = \frac{1}{N}\sum_{n=1}^{N} \mathrm{chrP}_n,
          \qquad
          \mathrm{chrR} = \frac{1}{N}\sum_{n=1}^{N} \mathrm{chrR}_n,

.. math:: \mathrm{chrF}\beta =
          (1 + \beta^2)\,\frac{\mathrm{chrP}\cdot\mathrm{chrR}}
                              {\beta^2\,\mathrm{chrP} + \mathrm{chrR}}

with :math:`N = 6` and :math:`\beta = 2` the paper's recommended
defaults -- :math:`\beta = 2` weights recall twice as heavily as
precision, which the paper found correlated best with human judgement.

Note the **arithmetic mean over orders**, not the geometric mean BLEU
uses. A geometric mean is zero as soon as any single order has no
match; chrF degrades smoothly instead, which is why it stays useful on
short segments.

Matching is clipped per n-gram type, as in ROUGE and BLEU: a character
n-gram present twice in the reference can match at most twice however
often the hypothesis repeats it.

Routes
------
``remove_whitespace`` (the paper's default, True) strips all spaces
before extracting character n-grams, so word boundaries do not
themselves become features.

``word_order`` > 0 adds the word n-gram terms of chrF++ (Popović 2017),
averaged in alongside the character terms.
"""

from ._richresult import RichResult

__all__ = ["chrf_score", "chrF"]


def _char_ngrams(s, n):
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _word_ngrams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def _counts(seq):
    d = {}
    for g in seq:
        d[g] = d.get(g, 0) + 1
    return d


def _pr(hyp_grams, ref_grams):
    """Clipped precision and recall for one n-gram order."""
    if not hyp_grams or not ref_grams:
        return None
    hc, rc = _counts(hyp_grams), _counts(ref_grams)
    match = sum(min(v, rc.get(g, 0)) for g, v in hc.items())
    return match / float(len(hyp_grams)), match / float(len(ref_grams))


def chrf_score(hypothesis, reference, n_char=6, beta=2.0,
               remove_whitespace=True, word_order=0):
    r"""chrF-beta between a hypothesis and one or more references."""
    N = int(n_char)
    if N < 1:
        raise ValueError("chrf_score: n_char must be at least 1, got %r"
                         % (n_char,))
    beta = float(beta)
    if beta <= 0.0:
        raise ValueError("chrf_score: beta must be positive, got %r" % (beta,))
    w_order = int(word_order)
    if w_order < 0:
        raise ValueError("chrf_score: word_order must be >= 0")

    refs = [reference] if isinstance(reference, str) else list(reference)
    hyp = str(hypothesis)

    best = None
    for ref in refs:
        ref = str(ref)
        h_words, r_words = hyp.split(), ref.split()
        h_chars = "".join(h_words) if remove_whitespace else hyp
        r_chars = "".join(r_words) if remove_whitespace else ref

        precs, recs = [], []
        per_order = []
        for n in range(1, N + 1):
            pr = _pr(_char_ngrams(h_chars, n), _char_ngrams(r_chars, n))
            # Orders with no n-grams on either side are SKIPPED, not
            # counted as zero: a 3-character segment has no 6-grams, and
            # scoring that as a miss would penalise shortness twice.
            if pr is None:
                per_order.append(None)
                continue
            precs.append(pr[0])
            recs.append(pr[1])
            per_order.append({"n": n, "precision": pr[0], "recall": pr[1]})

        for n in range(1, w_order + 1):
            pr = _pr(_word_ngrams(h_words, n), _word_ngrams(r_words, n))
            if pr is None:
                continue
            precs.append(pr[0])
            recs.append(pr[1])

        # ARITHMETIC mean over orders (not geometric, unlike BLEU).
        chrP = sum(precs) / len(precs) if precs else 0.0
        chrR = sum(recs) / len(recs) if recs else 0.0
        b2 = beta * beta
        denom = b2 * chrP + chrR
        f = (1.0 + b2) * chrP * chrR / denom if denom > 0.0 else 0.0
        cand = {"estimate": f, "chrf": f, "chrP": chrP, "chrR": chrR,
                "per_order": per_order, "n_char": N, "beta": beta,
                "word_order": w_order,
                "remove_whitespace": bool(remove_whitespace)}
        if best is None or cand["chrf"] > best["chrf"]:
            best = cand

    best["method"] = ("chrF%g, arithmetic mean over character n-gram orders "
                      "(Popovic 2015)" % beta)
    return RichResult(payload=best)


def cheatsheet():
    return ("chrF: chrP/chrR = ARITHMETIC mean of clipped char n-gram "
            "precision/recall over n=1..6; chrFb = (1+b^2) PR/(b^2 P + R), "
            "b=2; whitespace stripped; word_order>0 gives chrF++.")


chrF = chrf_score
