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

:math:`\lambda` and :math:`K` come from

    Karlin, S., & Altschul, S. F. (1990) "Methods for assessing the
    statistical significance of molecular sequence features by using general
    scoring schemes", *PNAS* 87(6), 2264-2268,

which the BLAST paper cites and does not reproduce. Both are computed in
closed form by :func:`karlin_altschul`. :math:`\lambda^*` is "the unique
positive solution to the equation"

.. math:: \sum_{i} p_i e^{\lambda s_i} = 1,

which needs "at least one score to be positive" and the expected score per
letter to be negative -- otherwise "the maximal segment would tend to be the
whole sequence, and this is not of interest". :math:`K^*` is the paper's
Appendix. With :math:`S_k` the sum of :math:`k` independently chosen scores,

.. math:: C^* = \frac{\exp\Big\{-2\sum_{k\ge1}\frac{1}{k}
          \big(E[e^{\lambda^* S_k}; S_k < 0] + \Pr(S_k \ge 0)\big)
          \Big\}}{\lambda^* E[S_1 e^{\lambda^* S_1}]},

and :math:`K^*` is bracketed by

.. math:: K^- = C^*\frac{\lambda^*\delta}{e^{\lambda^*\delta} - 1},
          \qquad
          K^+ = C^*\frac{\lambda^*\delta}{1 - e^{-\lambda^*\delta}},

where :math:`\delta` "is the smallest span of score values. When all scores
are integers with greatest common divisor 1, then :math:`\delta = 1`." The
upper bound is returned by default because "using :math:`K^+` for :math:`K^*`
always provides a conservative estimate of statistical significance", and the
series "converges geometrically fast, so that only a small number of terms
are needed".

:func:`estimate_gumbel` remains as a second route: simulate random pairs, take
the exact MSP of each, and regress :math:`\log(-\log \hat F(S))` on
:math:`S`, whose slope is :math:`-\lambda` and whose intercept is
:math:`\log(Kmn)`. It is useful when the score model is not a simple
independent-letter one; the anchor checks the two routes against each other.
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
          lam=None, K=None, max_hsps=None, letter_probs=None,
          pvalues=True):
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
        The Karlin-Altschul parameters. Omitted, they are computed in
        closed form by :func:`karlin_altschul` from this search's own
        scoring scheme and ``letter_probs``, so p-values come back by
        default. Pass them to override, or set ``pvalues=False`` to skip.
    letter_probs : sequence of float, optional
        Background letter frequencies for that calculation; uniform over
        ``alphabet`` when omitted.
    pvalues : bool
        Set ``False`` to report no p-values at all.
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

    ka = None
    if pvalues and (lam is None or K is None):
        try:
            ka = karlin_altschul(None, match, mismatch,
                                 letter_probs or [1.0 / len(alphabet)] *
                                 len(alphabet), matrix)
            lam = ka["lam"] if lam is None else lam
            K = ka["K"] if K is None else K
        except ValueError:
            ka = None
    if pvalues and lam is not None and K is not None:
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
        "lam": lam,
        "K": K,
        "karlin_altschul": ka,
        "note": "the X-drop extension is a heuristic: it may miss a "
                "higher-scoring extension (Altschul et al. 1990, section "
                "2c); msp_exact gives the guaranteed MSP score",
        "method": "BLAST maximal segment pairs (Altschul et al. 1990)",
    })


def _lattice(x):
    """Integer score, or a clear error.

    Karlin & Altschul's theory is stated for scores on a lattice, and the
    module works on the unit lattice (delta is their gcd). Silently
    truncating a non-integer score changes the scoring scheme -- -0.01
    would become 0 and the expected score per letter would go positive --
    so rescale the scheme yourself rather than have it done behind your
    back.
    """
    v = float(x)
    n = int(round(v))
    if abs(v - n) > 1e-9:
        raise ValueError("blstn: scores must lie on the integer lattice "
                         "(got %r); multiply the whole scheme by a common "
                         "factor first" % (x,))
    return n


def score_distribution(match=5, mismatch=-4, letter_probs=None,
                       matrix=None, subject_probs=None):
    r"""The distribution of a single aligned-pair score.

    For DNA-style scoring, :math:`P(s) = \sum_{i,j: s_{ij} = s} p_i q_j`
    with match and mismatch scores; for a substitution matrix, the same sum
    over its entries. Returns ``{score: probability}`` with integer scores.
    """
    if matrix is None:
        p = list(letter_probs) if letter_probs else [0.25] * 4
        tot = sum(p)
        p = [v / tot for v in p]
        q = p if subject_probs is None else [float(v) for v in subject_probs]
        qt = sum(q)
        q = [v / qt for v in q]
        out = {}
        for i in range(len(p)):
            for j in range(len(q)):
                sc = _lattice(match) if i == j else _lattice(mismatch)
                out[sc] = out.get(sc, 0.0) + p[i] * q[j]
        return out
    p = list(letter_probs)
    q = p if subject_probs is None else list(subject_probs)
    out = {}
    for i in range(len(p)):
        for j in range(len(q)):
            sc = _lattice(matrix[i][j])
            out[sc] = out.get(sc, 0.0) + p[i] * q[j]
    return out


def _lambda_star(dist, hi=20.0, tol=1e-14, max_iter=300):
    r"""The unique positive root of :math:`\sum_i p_i e^{\lambda s_i} = 1`."""
    scores = sorted(dist)
    mean = sum(s * dist[s] for s in scores)
    if mean >= 0:
        raise ValueError("blstn: the expected score per letter must be "
                         "negative (it is %.6g); otherwise the maximal "
                         "segment is the whole sequence" % mean)
    if max(scores) <= 0:
        raise ValueError("blstn: at least one score must be positive")

    def f(lam):
        return sum(dist[s] * math.exp(lam * s) for s in scores) - 1.0

    lo = 1e-12
    while f(hi) < 0 and hi < 700:
        hi *= 2.0
    if f(hi) < 0:
        raise ValueError("blstn: no positive root for lambda was found")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _gcd_span(scores):
    """delta: the smallest span of score values (gcd of the scores)."""
    g = 0
    for s in scores:
        a = abs(int(s))
        while a:
            g, a = a, g % a
    return g if g > 0 else 1


def karlin_altschul(dist=None, match=5, mismatch=-4, letter_probs=None,
                    matrix=None, subject_probs=None, max_terms=1000,
                    tol=1e-12, bound="upper"):
    r"""Karlin & Altschul (1990): :math:`\lambda^*` and :math:`K^*`.

    Give either a score distribution ``dist`` (``{score: probability}``) or
    the ingredients for :func:`score_distribution`.

    The series is truncated when a term falls below ``tol``; for +5/-4 DNA
    that is 226 terms, and the tail beyond 200 moves :math:`K` by less than
    :math:`10^{-8}`, so ``max_terms`` exists as a guard rather than as a
    tuning knob.

    ``bound`` selects which of the Appendix's brackets is returned as ``K``:
    ``"upper"`` (:math:`K^+`, the paper's conservative choice),
    ``"lower"`` (:math:`K^-`) or ``"mid"`` (their mean). All three are in
    the result either way.
    """
    if dist is None:
        dist = score_distribution(match, mismatch, letter_probs, matrix,
                                  subject_probs)
    dist = dict((_lattice(s), float(p)) for s, p in dist.items() if p > 0)
    tot = sum(dist.values())
    dist = dict((s, p / tot) for s, p in dist.items())
    if bound not in ("upper", "lower", "mid"):
        raise ValueError("blstn: bound must be 'upper', 'lower' or 'mid'")
    lam = _lambda_star(dist)
    delta = _gcd_span(dist)

    # E[S_1 e^{lambda S_1}]
    denom = lam * sum(s * dist[s] * math.exp(lam * s) for s in dist)
    if denom <= 0:
        raise ValueError("blstn: the normalising expectation is not "
                         "positive; check the scoring scheme")

    # the Appendix series, over the convolutions S_k
    conv = dict(dist)
    series = 0.0
    terms = []
    for k in range(1, int(max_terms) + 1):
        e_neg = sum(conv[s] * math.exp(lam * s) for s in conv if s < 0)
        p_ge0 = sum(conv[s] for s in conv if s >= 0)
        term = (e_neg + p_ge0) / k
        series += term
        terms.append(term)
        if term < tol and k > 5:
            break
        nxt = {}
        for a, pa in conv.items():
            for b, pb in dist.items():
                nxt[a + b] = nxt.get(a + b, 0.0) + pa * pb
        # drop negligible far-negative tail to keep the convolution finite
        conv = dict((s, p) for s, p in nxt.items()
                    if p * math.exp(lam * min(s, 0)) > 1e-300 and p > 1e-300)
    c_star = math.exp(-2.0 * series) / denom
    x = lam * delta
    k_low = c_star * x / (math.exp(x) - 1.0)
    k_high = c_star * x / (1.0 - math.exp(-x))
    k = {"upper": k_high, "lower": k_low,
         "mid": 0.5 * (k_low + k_high)}[bound]
    return {"lam": lam, "K": k, "K_upper": k_high, "K_lower": k_low,
            "C": c_star, "delta": delta, "terms": len(terms),
            "series": series,
            "mean_score": sum(s * dist[s] for s in dist),
            "distribution": dist}


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
            "eq.2 for c or more distinct segment pairs. lambda and K come "
            "from Karlin & Altschul (1990) in closed form: lambda solves "
            "sum p_i e^{lambda s_i} = 1, and K from the Appendix series "
            "C* with the bracket K- <= K <= K+, K+ being the conservative "
            "choice. estimate_gumbel is the simulation alternative.")


# compact alias per ledger/NAMING.md
blast = blstn

# name carried over from the generated stub this replaced
blast_nucleotide = blstn
