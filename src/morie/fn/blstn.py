r"""BLAST: local alignment by maximal segment pairs.

Altschul, S. F., Gish, W., Miller, W., Myers, E. W., & Lipman, D. J. (1990)
"Basic Local Alignment Search Tool", *Journal of Molecular Biology* 215,
403-410.

The measure is the **maximal segment pair** (MSP): "the highest scoring pair
of identical length segments chosen from 2 sequences", with boundaries chosen
to maximise the score, so an MSP may be of any length. A segment pair is
**locally maximal** if "its score cannot be improved either by extending or by
shortening both segments" (Sellers 1984), and BLAST seeks all locally maximal
segment pairs scoring above a cutoff :math:`S`.

The MSP score can be computed exactly "in time proportional to the product of
their lengths using a simple dynamic programming algorithm"; that exact route
is implemented here as :func:`msp_exact` and is what the heuristic is measured
against. The heuristic itself is three steps:

1. compile a word list -- for DNA "the list of all contiguous
   :math:`w`-mers in the query sequence, often with :math:`w = 12`"; for
   proteins "all words (:math:`w`-mers) that score at least :math:`T` when
   compared to some word in the query sequence";
2. scan the subject for hits, using the paper's own device of indexing every
   word to an integer and looking up its occurrences;
3. extend each hit in both directions, stopping "when we reach a segment pair
   whose score falls a certain distance below the best score found for shorter
   extensions" -- the X-drop, "for protein comparisons the default distance
   is 20".

Both word routes are implemented (``word_mode``), because they are two
different algorithms with different sensitivity, and the paper gives both.
The X-drop is the paper's stated departure from guaranteed MSPs: "this
introduces a further departure from the ideal of finding guaranteed MSPs, but
the added inaccuracy is negligible ... the probability of missing a higher
scoring extension is about 0.001". Lowering :math:`T` or :math:`w`, or raising
:math:`X`, moves the method back toward the exact answer at a cost in time.

**Significance.** For two random sequences of lengths :math:`m` and :math:`n`,
the probability of a segment pair scoring at least :math:`S` is

.. math:: 1 - e^{-y}, \qquad y = K m n e^{-\lambda S}   \tag{1}

and the probability of :math:`c` or more distinct segment pairs all scoring at
least :math:`S` is

.. math:: 1 - e^{-y} \sum_{i=0}^{c-1} \frac{y^i}{i!}.   \tag{2}

Equation 2 is what lets "two sequences that share several distinct regions of
similarity ... sometimes be detected as significantly related, even when no
segment pair is statistically significant in isolation".

:math:`\lambda` and :math:`K` come from Karlin & Altschul (1990), which the
paper cites for them and does not reproduce. That paper is not in this
library, so no closed form for either constant is claimed here. Instead
:func:`estimate_gumbel` estimates both **from equation 1 itself** -- simulate
random sequence pairs under the given letter frequencies and scoring scheme,
take the MSP of each, and regress :math:`\log(-\log \hat F(S))` on :math:`S`,
whose slope is :math:`-\lambda` and whose intercept is :math:`\log(Kmn)`.
That is an estimator this module supplies, not a result of the paper; pass
``lam`` and ``K`` directly when you have the published constants for your
scoring scheme.
"""

import math

from ._richresult import RichResult

__all__ = ["blstn", "blast_nucleotide", "blast", "msp_exact", "blast_pvalue", "estimate_gumbel",
           "word_hits"]


def _scorer(match, mismatch, matrix, alphabet):
    if matrix is None:
        def f(a, b):
            return match if a == b else mismatch
    else:
        idx = dict((c, k) for k, c in enumerate(alphabet))

        def f(a, b):
            return matrix[idx[a]][idx[b]]
    return f


def msp_exact(query, subject, match=5, mismatch=-4, matrix=None,
              alphabet="ACGT"):
    r"""The exact maximal segment pair score, by dynamic programming.

    Every segment pair lies on one diagonal, so the best ungapped segment
    pair is the maximum-scoring contiguous run over all diagonals -- Kadane's
    scan on each -- in :math:`O(mn)`, which is the cost the paper quotes for
    computing the MSP exactly.

    Returns ``(score, qstart, sstart, length)``; the segment pair is
    ``query[qstart:qstart+length]`` against
    ``subject[sstart:sstart+length]``.
    """
    q, s = str(query), str(subject)
    sc = _scorer(match, mismatch, matrix, alphabet)
    best = (0, 0, 0, 0)
    for d in range(-(len(q) - 1), len(s)):
        qi = max(0, -d)
        si = max(0, d)
        run = 0.0
        run_start = 0
        t = 0
        while qi + t < len(q) and si + t < len(s):
            v = sc(q[qi + t], s[si + t])
            if run <= 0:
                run = v
                run_start = t
            else:
                run += v
            if run > best[0]:
                best = (run, qi + run_start, si + run_start,
                        t - run_start + 1)
            t += 1
    return best


def word_hits(query, subject, w, mode="exact", threshold=None, match=5,
              mismatch=-4, matrix=None, alphabet="ACGT"):
    r"""Hits between the query word list and the subject (step 2).

    ``mode="exact"`` is the DNA route: the word list is every contiguous
    :math:`w`-mer of the query and a hit is an exact occurrence.
    ``mode="neighborhood"`` is the protein route: the list is every
    :math:`w`-mer scoring at least ``threshold`` against some query word, so
    a hit need not be an identity.

    Returns a list of ``(qpos, spos)``.
    """
    q, s = str(query), str(subject)
    w = int(w)
    if w < 1:
        raise ValueError("blstn: w must be >= 1")
    if len(q) < w or len(s) < w:
        return []
    if mode not in ("exact", "neighborhood"):
        raise ValueError("blstn: mode must be 'exact' or 'neighborhood'")
    table = {}
    for i in range(len(q) - w + 1):
        table.setdefault(q[i:i + w], []).append(i)
    if mode == "exact":
        return [(i, j) for j in range(len(s) - w + 1)
                for i in table.get(s[j:j + w], ())]
    if threshold is None:
        raise ValueError("blstn: mode='neighborhood' needs a threshold T")
    sc = _scorer(match, mismatch, matrix, alphabet)
    hits = []
    for j in range(len(s) - w + 1):
        word = s[j:j + w]
        for qword, positions in table.items():
            tot = 0.0
            for t in range(w):
                tot += sc(qword[t], word[t])
            if tot >= threshold:
                for i in positions:
                    hits.append((i, j))
    return hits


def _extend(q, s, qi, si, w, sc, X):
    """Ungapped X-drop extension of a hit, both directions."""
    score = 0.0
    for t in range(w):
        score += sc(q[qi + t], s[si + t])
    # right
    run, cur = score, score
    best_right = 0
    t = 0
    while qi + w + t < len(q) and si + w + t < len(s):
        cur += sc(q[qi + w + t], s[si + w + t])
        t += 1
        if cur > run:
            run, best_right = cur, t
        elif run - cur > X:
            break
    # left
    cur = run
    best_left = 0
    t = 1
    while qi - t >= 0 and si - t >= 0:
        cur += sc(q[qi - t], s[si - t])
        if cur > run:
            run, best_left = cur, t
        elif run - cur > X:
            break
        t += 1
    best = run
    qs = qi - best_left
    ss = si - best_left
    length = w + best_right + best_left
    return best, qs, ss, length


def blstn(query, subjects, w=11, match=5, mismatch=-4, cutoff=None, X=20,
          word_mode="exact", threshold=None, matrix=None, alphabet="ACGT",
          lam=None, K=None, max_hsps=None):
    r"""Search ``subjects`` for locally maximal segment pairs with ``query``.

    Parameters
    ----------
    query : str
        The query sequence.
    subjects : str or sequence of str
        One subject or a database of them.
    w : int
        Word length. The paper uses :math:`w = 12` for DNA and small
        :math:`w` with a threshold for proteins.
    match, mismatch : float
        DNA scores; the paper's defaults are :math:`+5` and :math:`-4`,
        which are the defaults here. Ignored when ``matrix`` is given.
    cutoff : float, optional
        The score cutoff :math:`S`. Defaults to the score of a perfect
        :math:`w`-mer, so every reported segment pair is at least as good as
        the hit that seeded it.
    X : float
        The X-drop: extension stops when the running score falls this far
        below the best seen. The paper's protein default is 20.
    word_mode : {"exact", "neighborhood"}
        Which word list to compile; see :func:`word_hits`.
    threshold : float, optional
        :math:`T`, required by ``word_mode="neighborhood"``.
    matrix : list of list of float, optional
        A substitution matrix over ``alphabet`` (e.g. PAM-120), used instead
        of match/mismatch.
    alphabet : str
        Letter order for ``matrix``.
    lam, K : float, optional
        The Karlin-Altschul parameters. Supplied, the result carries
        p-values from equations 1 and 2; omitted, it does not, because
        guessing them would make the p-values fiction. See
        :func:`estimate_gumbel`.
    max_hsps : int, optional
        Keep only this many segment pairs per subject, highest score first.

    Returns
    -------
    RichResult
        ``estimate`` / ``hsps`` is a list of dicts with ``subject``,
        ``score``, ``qstart``, ``sstart``, ``length``, ``identities`` and
        (when ``lam`` and ``K`` are given) ``pvalue``; ``best_score`` is the
        heuristic MSP score, ``n_hits`` the number of word hits examined.

    Examples
    --------
    A planted identical region is found and scored::

        q = "ACGTACGTTTGACCAGGTAAC"
        s = "TTTTTTACGTACGTTTGACCAGGTAACGGG"
        blstn(q, s, w=8)["best_score"]      # 5 * 21 = 105

    The exact answer is available for comparison::

        msp_exact(q, s)[0]                  # 105

    References
    ----------
    Altschul, Gish, Miller, Myers & Lipman (1990) *J. Mol. Biol.* 215,
    403-410, sections 2(a)-(c) and equations 1-2.
    """
    q = str(query)
    if not q:
        raise ValueError("blstn: query must be non-empty")
    subs = [subjects] if isinstance(subjects, str) else [str(x) for x in
                                                        subjects]
    if not subs:
        raise ValueError("blstn: subjects must be non-empty")
    w = int(w)
    if w < 1:
        raise ValueError("blstn: w must be >= 1")
    X = float(X)
    if X < 0:
        raise ValueError("blstn: X must be >= 0")
    sc = _scorer(match, mismatch, matrix, alphabet)
    if cutoff is None:
        cutoff = w * (match if matrix is None else
                      max(matrix[k][k] for k in range(len(matrix))))
    cutoff = float(cutoff)

    hsps = []
    n_hits = 0
    for si, s in enumerate(subs):
        hits = word_hits(q, s, w, word_mode, threshold, match, mismatch,
                         matrix, alphabet)
        n_hits += len(hits)
        seen = set()
        for (qi, sj) in hits:
            score, qs, ss, length = _extend(q, s, qi, sj, w, sc, X)
            key = (si, qs - ss, qs, length)
            if key in seen or score < cutoff:
                continue
            seen.add(key)
            ident = sum(1 for t in range(length)
                        if q[qs + t] == s[ss + t])
            hsps.append({"subject": si, "score": score, "qstart": qs,
                         "sstart": ss, "length": length,
                         "identities": ident})
    # locally maximal: drop a segment pair contained in a better one on the
    # same diagonal
    hsps.sort(key=lambda h: -h["score"])
    kept = []
    for h in hsps:
        d = h["qstart"] - h["sstart"]
        covered = False
        for g in kept:
            if (g["subject"] == h["subject"] and
                    g["qstart"] - g["sstart"] == d and
                    g["qstart"] <= h["qstart"] and
                    h["qstart"] + h["length"] <=
                    g["qstart"] + g["length"]):
                covered = True
                break
        if not covered:
            kept.append(h)
    if max_hsps is not None:
        kept = kept[:int(max_hsps)]

    if lam is not None and K is not None:
        for h in kept:
            m = len(subs[h["subject"]])
            h["pvalue"] = blast_pvalue(h["score"], len(q), m, lam, K)
    return RichResult(payload={
        "estimate": kept,
        "hsps": kept,
        "best_score": max([h["score"] for h in kept] or [0.0]),
        "n_hsps": len(kept),
        "n_hits": n_hits,
        "w": w,
        "cutoff": cutoff,
        "X": X,
        "word_mode": word_mode,
        "note": "the X-drop extension is a heuristic: it may miss a "
                "higher-scoring extension (Altschul et al. 1990, section "
                "2c); msp_exact gives the guaranteed MSP score",
        "method": "BLAST maximal segment pairs (Altschul et al. 1990)",
    })


def blast_pvalue(score, m, n, lam, K, c=1):
    r"""Equations 1 and 2 of Altschul et al. (1990).

    The probability that random sequences of lengths :math:`m` and :math:`n`
    contain at least ``c`` distinct segment pairs scoring ``score`` or more:
    :math:`1 - e^{-y}\sum_{i<c} y^i/i!` with
    :math:`y = K m n e^{-\lambda S}`. ``c = 1`` is equation 1.
    """
    lam = float(lam)
    K = float(K)
    if lam <= 0 or K <= 0:
        raise ValueError("blstn: lam and K must be positive")
    c = int(c)
    if c < 1:
        raise ValueError("blstn: c must be >= 1")
    y = K * float(m) * float(n) * math.exp(-lam * float(score))
    tail = 0.0
    term = 1.0
    for i in range(c):
        if i:
            term *= y / i
        tail += term
    p = 1.0 - math.exp(-y) * tail
    return min(max(p, 0.0), 1.0)


def estimate_gumbel(m, n, letter_freqs, match=5, mismatch=-4, matrix=None,
                    alphabet="ACGT", n_sim=200, seed=0, quantiles=(0.2, 0.9)):
    r"""Estimate :math:`\lambda` and :math:`K` by simulation from equation 1.

    Karlin & Altschul (1990) give closed forms; that paper is not in this
    library, so this is an estimator rather than their formula. Equation 1
    says :math:`\Pr(S < x) = \exp(-Kmn e^{-\lambda x})`, hence

    .. math:: \log(-\log \hat F(x)) = \log(Kmn) - \lambda x,

    a straight line. Simulate ``n_sim`` random sequence pairs of lengths
    ``m`` and ``n`` from ``letter_freqs``, take the exact MSP of each, and fit
    that line over the central ``quantiles`` of the empirical distribution.

    Returns ``{"lam": ..., "K": ..., "scores": [...]}``.
    """
    freqs = [float(x) for x in letter_freqs]
    tot = sum(freqs)
    if tot <= 0:
        raise ValueError("blstn: letter_freqs must be positive")
    freqs = [x / tot for x in freqs]
    cum = []
    run = 0.0
    for x in freqs:
        run += x
        cum.append(run)
    # a plain LCG, taken from the high bits: the estimator only needs an
    # even spread and the module carries no external RNG dependency
    state = [int(seed) or 1]

    def rnd():
        state[0] = (1103515245 * state[0] + 12345) % (1 << 31)
        return state[0] / float(1 << 31)

    def draw(length):
        out = []
        for _ in range(length):
            u = rnd()
            k = 0
            while k < len(cum) - 1 and u > cum[k]:
                k += 1
            out.append(alphabet[k])
        return "".join(out)

    scores = []
    for _ in range(int(n_sim)):
        a = draw(int(m))
        b = draw(int(n))
        scores.append(msp_exact(a, b, match, mismatch, matrix, alphabet)[0])
    scores.sort()
    N = len(scores)
    lo = max(1, int(quantiles[0] * N))
    hi = min(N - 1, int(quantiles[1] * N))
    xs, ys = [], []
    for r in range(lo, hi + 1):
        F = r / float(N + 1)
        if 0.0 < F < 1.0:
            xs.append(scores[r - 1])
            ys.append(math.log(-math.log(F)))
    if len(xs) < 2:
        raise ValueError("blstn: too few distinct simulated scores to fit; "
                         "raise n_sim")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    if sxx <= 0:
        raise ValueError("blstn: simulated scores are constant; raise n_sim")
    slope = sxy / sxx
    intercept = my - slope * mx
    lam = -slope
    if lam <= 0:
        raise ValueError("blstn: the fitted lambda is not positive -- the "
                         "expected score per pair is probably non-negative, "
                         "which breaks the local-alignment model")
    return {"lam": lam, "K": math.exp(intercept) / (float(m) * float(n)),
            "scores": scores}


def cheatsheet():
    return ("blstn: BLAST (Altschul et al. 1990). The measure is the "
            "MAXIMAL SEGMENT PAIR -- the best-scoring equal-length "
            "ungapped segment pair, boundaries chosen to maximise the "
            "score. Exact by DP in O(mn) (msp_exact); the heuristic seeds "
            "on word hits (all contiguous w-mers for DNA, or all w-mers "
            "scoring >= T against a query word for protein) and extends "
            "each hit until the score falls X below the best seen, so it "
            "can MISS the true MSP -- that is the trade, not a bug. "
            "Significance from eq.1, 1 - exp(-Kmn e^{-lambda S}), and "
            "eq.2 for c or more distinct segment pairs. lambda and K are "
            "Karlin-Altschul constants: pass them, or estimate them from "
            "eq.1 by simulation with estimate_gumbel.")


# compact alias per ledger/NAMING.md
blast = blstn

# name carried over from the generated stub this replaced
blast_nucleotide = blstn
