r"""ROUGE: recall-oriented understudy for gisting evaluation.

Lin, C.-Y. (2004) "ROUGE: A Package for Automatic Evaluation of
Summaries", *Text Summarization Branches Out (ACL workshop)*, 74-81.

Three variants, all in the paper and all implemented here.

**ROUGE-N** (Sec. 2), n-gram recall against the reference:

.. math:: \mathrm{ROUGE\text{-}N} =
    \frac{\sum_{S \in \{Ref\}} \sum_{g_n \in S}
          \mathrm{Count}_{match}(g_n)}
         {\sum_{S \in \{Ref\}} \sum_{g_n \in S} \mathrm{Count}(g_n)}

where :math:`\mathrm{Count}_{match}` is **clipped** -- an n-gram
occurring twice in the reference and five times in the candidate counts
twice, not five times. Without the clip, repeating one good word
inflates the score without bound.

**ROUGE-L** (Sec. 3), longest common subsequence, which rewards
in-order overlap without requiring contiguity:

.. math:: R_{lcs} = \frac{LCS(X,Y)}{m}, \quad
          P_{lcs} = \frac{LCS(X,Y)}{n}, \quad
          F_{lcs} = \frac{(1+\beta^2) R_{lcs} P_{lcs}}
                         {R_{lcs} + \beta^2 P_{lcs}}

with :math:`m, n` the reference and candidate lengths.

**ROUGE-W** (Sec. 4), weighted LCS, which additionally prefers
*consecutive* matches by scoring a run of length :math:`k` as
:math:`f(k) = k^\alpha` rather than :math:`k`. Two candidates with the
same LCS length but different contiguity get the same ROUGE-L and
different ROUGE-W; that difference is the reason the variant exists.
The inverse :math:`f^{-1}(x) = x^{1/\alpha}` is applied before forming
the ratios, exactly as in the paper.

Multiple references are handled per Sec. 3.2 by taking the best score
over references.
"""

from ._richresult import RichResult

__all__ = ["rouge_n", "rouge_l", "rouge_w", "rouge", "lcs_length"]


def _toks(x):
    if isinstance(x, str):
        return x.split()
    return [str(t) for t in x]


def _ngrams(toks, n):
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _counts(seq):
    d = {}
    for g in seq:
        d[g] = d.get(g, 0) + 1
    return d


def _prf(match, n_cand, n_ref, beta):
    p = match / float(n_cand) if n_cand else 0.0
    r = match / float(n_ref) if n_ref else 0.0
    if p <= 0.0 or r <= 0.0:
        f = 0.0
    else:
        b2 = beta * beta
        f = (1.0 + b2) * p * r / (r + b2 * p)
    return p, r, f


def rouge_n(candidate, reference, n=1, beta=1.0):
    r"""ROUGE-N with clipped n-gram matching."""
    n = int(n)
    if n < 1:
        raise ValueError("rouge_n: n must be at least 1, got %r" % (n,))
    c = _ngrams(_toks(candidate), n)
    refs = reference if (isinstance(reference, (list, tuple))
                         and reference and not isinstance(reference[0], str)
                         ) else [reference]
    if isinstance(reference, (list, tuple)) and reference and \
            isinstance(reference[0], str) and \
            all(" " in r or len(_toks(r)) > 1 for r in reference):
        refs = list(reference)
    best = None
    for ref in refs:
        rg = _ngrams(_toks(ref), n)
        cc, rc = _counts(c), _counts(rg)
        # CLIPPED: min of the two counts, per Sec. 2.
        match = sum(min(v, rc.get(g, 0)) for g, v in cc.items())
        p, r, f = _prf(match, len(c), len(rg), beta)
        cand = {"precision": p, "recall": r, "f1": f, "matches": match,
                "n_candidate": len(c), "n_reference": len(rg)}
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    best["estimate"] = best["recall"]        # ROUGE-N is recall-oriented
    best["n"] = n
    best["method"] = "ROUGE-N, clipped n-gram recall (Lin 2004, Sec. 2)"
    return RichResult(payload=best)


def lcs_length(a, b):
    """Length of the longest common subsequence (dynamic programming)."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[n]


def rouge_l(candidate, reference, beta=1.0):
    r"""ROUGE-L, LCS-based F measure (Lin 2004, Sec. 3)."""
    c = _toks(candidate)
    refs = [reference] if isinstance(reference, str) else list(reference)
    best = None
    for ref in refs:
        rt = _toks(ref)
        l = lcs_length(c, rt)
        p, r, f = _prf(l, len(c), len(rt), beta)
        cand = {"precision": p, "recall": r, "f1": f, "lcs": l,
                "n_candidate": len(c), "n_reference": len(rt)}
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    best["estimate"] = best["f1"]
    best["beta"] = beta
    best["method"] = "ROUGE-L, LCS-based F measure (Lin 2004, Sec. 3)"
    return RichResult(payload=best)


def _wlcs(a, b, alpha):
    """Weighted LCS: a run of k consecutive matches scores k^alpha."""
    m, n = len(a), len(b)
    c = [[0.0] * (n + 1) for _ in range(m + 1)]
    w = [[0] * (n + 1) for _ in range(m + 1)]   # current run length
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                k = w[i - 1][j - 1]
                # Extending a run of k to k+1 adds f(k+1) - f(k), so a
                # long consecutive block is worth more than the same
                # number of scattered matches.
                c[i][j] = c[i - 1][j - 1] + ((k + 1.0) ** alpha
                                             - float(k) ** alpha)
                w[i][j] = k + 1
            else:
                if c[i - 1][j] >= c[i][j - 1]:
                    c[i][j] = c[i - 1][j]
                else:
                    c[i][j] = c[i][j - 1]
                w[i][j] = 0
    return c[m][n]


def rouge_w(candidate, reference, alpha=1.2, beta=1.0):
    r"""ROUGE-W, weighted LCS favouring consecutive matches (Sec. 4)."""
    alpha = float(alpha)
    if alpha < 1.0:
        raise ValueError(
            "rouge_w: alpha must be at least 1 for consecutive matches to "
            "be preferred, got %r" % (alpha,))
    c = _toks(candidate)
    refs = [reference] if isinstance(reference, str) else list(reference)
    best = None
    for ref in refs:
        rt = _toks(ref)
        wl = _wlcs(c, rt, alpha)
        inv = 1.0 / alpha
        # f^-1 applied before the ratios, per Sec. 4.
        p = (wl / (len(c) ** alpha)) ** inv if c else 0.0
        r = (wl / (len(rt) ** alpha)) ** inv if rt else 0.0
        if p <= 0.0 or r <= 0.0:
            f = 0.0
        else:
            b2 = beta * beta
            f = (1.0 + b2) * p * r / (r + b2 * p)
        cand = {"precision": p, "recall": r, "f1": f, "wlcs": wl,
                "n_candidate": len(c), "n_reference": len(rt)}
        if best is None or cand["f1"] > best["f1"]:
            best = cand
    best["estimate"] = best["f1"]
    best["alpha"] = alpha
    best["method"] = "ROUGE-W, weighted LCS (Lin 2004, Sec. 4)"
    return RichResult(payload=best)


def rouge(candidate, reference, variant="L", n=1, alpha=1.2, beta=1.0):
    """Dispatch to ROUGE-N, ROUGE-L or ROUGE-W."""
    v = str(variant).upper()
    if v == "L":
        return rouge_l(candidate, reference, beta=beta)
    if v == "W":
        return rouge_w(candidate, reference, alpha=alpha, beta=beta)
    if v == "N":
        return rouge_n(candidate, reference, n=n, beta=beta)
    raise ValueError("rouge: variant must be N, L or W, got %r" % (variant,))


def cheatsheet():
    return ("rouge: ROUGE-N clipped n-gram recall; ROUGE-L LCS F with "
            "R=LCS/m, P=LCS/n; ROUGE-W weighted LCS f(k)=k^alpha with "
            "f^-1 before the ratios; best over multiple references.")
