r"""RAPPOR: randomized aggregatable privacy-preserving ordinal response.

Erlingsson, Ú., Pihur, V., & Korolova, A. (2014) "RAPPOR: Randomized
Aggregatable Privacy-Preserving Ordinal Response", *Proceedings of the
2014 ACM SIGSAC Conference on Computer and Communications Security
(CCS '14)*, 1054-1067, doi:10.1145/2660267.2660348.

Bloom, B. H. (1970) "Space/time trade-offs in hash coding with allowable
errors", *Communications of the ACM* 13(7), 422-426,
doi:10.1145/362686.362692 -- the filter built in step 1, and the source
of the false-positive rate that sets the k and h defaults.

The client never sends its value. It sends a doubly-randomized bit
array, and the server recovers *population* counts from many such
arrays. Section 2, the four steps:

1. **Signal.** Hash the value :math:`v` into a Bloom filter :math:`B` of
   size :math:`k` with :math:`h` hash functions.
2. **Permanent randomized response.** For each bit,

   .. math:: B'_i = \begin{cases}
             1 & \text{with probability } \tfrac12 f\\
             0 & \text{with probability } \tfrac12 f\\
             B_i & \text{with probability } 1 - f
             \end{cases}

   :math:`B'` is **memoized** and reused for every future report on this
   value. That is the whole longitudinal defence: without it, an
   attacker averaging many reports recovers :math:`B` exactly.
3. **Instantaneous randomized response.**
   :math:`P(S_i = 1) = q` if :math:`B'_i = 1`, else :math:`p`.
4. **Report** :math:`S`.

Privacy (Theorems 1 and 2), both reported by :func:`rappor_epsilon`:

.. math:: \varepsilon_\infty = 2h \ln\frac{1 - \tfrac12 f}{\tfrac12 f},
          \qquad
          \varepsilon_1 = h \log\frac{q^*(1-p^*)}{p^*(1-q^*)},

with Lemma 1's

.. math:: q^* = \tfrac12 f(p+q) + (1-f)q, \qquad
          p^* = \tfrac12 f(p+q) + (1-f)p.

:math:`\varepsilon_\infty` is the bound against an attacker who has seen
*infinitely many* reports and therefore knows :math:`B'`; it depends
only on :math:`f`, which is why :math:`f` is the longitudinal knob.

Decoding (Section 4). Since a set bit reports 1 with probability
:math:`q^*` and an unset bit with :math:`p^*`, the count :math:`c_{ij}`
of ones at bit :math:`i` in cohort :math:`j` over :math:`N_j` reports
estimates the true number of set bits as

.. math:: t_{ij} = \frac{c_{ij} - (p + \tfrac12 fq - \tfrac12 fp)N_j}
                        {(1-f)(q-p)},

which is the paper's printed estimator and equals
:math:`(c_{ij} - p^* N_j)/(q^* - p^*)`. It is unbiased by construction,
and *not* clipped to :math:`[0, N_j]` here: a negative estimate is
honest evidence that a bit is rare, and clipping it silently biases the
aggregate upward.

Routes
------
Section 2.1 lists three variants of the algorithm, all implemented and
selected by ``variant``:

``"full"``
    Bloom filter, PRR, then IRR. The default and the deployed scheme.
``"one-time"``
    Skips step 3. Appropriate when the client guarantees a single
    collection, so no longitudinal defence is needed; the report is
    :math:`B'` itself.
``"basic"``
    No Bloom filter -- a one-hot array over the candidate strings, so
    :math:`k` is the alphabet size and :math:`h = 1`. Avoids hash
    collisions entirely, at the cost of a report as wide as the
    alphabet. Better for small alphabets and small samples.
"""

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["rappor_encode", "rappor_decode", "rappor_epsilon",
           "rappor_star_probs", "rappor"]

_VARIANTS = ("full", "one-time", "basic")


def _check_params(f, p, q):
    if not (0.0 <= f <= 1.0):
        raise ValueError("rappor: f must lie in [0, 1], got %r" % (f,))
    if not (0.0 <= p <= 1.0):
        raise ValueError("rappor: p must lie in [0, 1], got %r" % (p,))
    if not (0.0 <= q <= 1.0):
        raise ValueError("rappor: q must lie in [0, 1], got %r" % (q,))


def rappor_star_probs(f, p, q):
    """Lemma 1: the effective per-bit report probabilities."""
    f, p, q = float(f), float(p), float(q)
    _check_params(f, p, q)
    half = 0.5 * f * (p + q)
    return half + (1.0 - f) * q, half + (1.0 - f) * p


def rappor_epsilon(h, f, p=None, q=None):
    r"""Theorem 1 and, when ``p``/``q`` are given, Theorem 2."""
    h = int(h)
    if h < 1:
        raise ValueError("rappor: h must be at least 1, got %r" % (h,))
    f = float(f)
    if not (0.0 < f < 2.0):
        raise ValueError(
            "rappor: f must lie in (0, 2) for eps_inf to be finite, got %r"
            % (f,))
    half_f = 0.5 * f
    eps_inf = 2.0 * h * math.log((1.0 - half_f) / half_f)
    out = {"eps_infinity": eps_inf}
    if p is not None and q is not None:
        qs, ps = rappor_star_probs(f, p, q)
        if not (0.0 < ps < 1.0) or not (0.0 < qs < 1.0):
            raise ValueError(
                "rappor: q* and p* must lie strictly in (0, 1); got "
                "q*=%r p*=%r" % (qs, ps))
        out["eps_1"] = h * math.log((qs * (1.0 - ps)) / (ps * (1.0 - qs)))
        out["q_star"] = qs
        out["p_star"] = ps
    return out


def _bloom(value, k, h, cohort=0):
    """h hash positions for `value`, deterministic and cohort-specific.

    The paper assigns each client a cohort with its own hash functions
    so that collisions differ across cohorts; a string unlucky in one
    cohort is recoverable from the others.
    """
    bits = []
    s = str(value)
    for j in range(h):
        # A polynomial rolling hash mod (2^31 - 1), seeded by the cohort
        # and the hash index. Deliberately 31-bit and multiplicative by
        # 131: every intermediate stays under 2^38, so it is EXACT in
        # IEEE doubles and the R arm reproduces it bit for bit. A 64-bit
        # FNV would not -- R has no exact 64-bit integer arithmetic.
        acc = (cohort * 7919 + j * 104729 + 1) % 2147483647
        for ch in s:
            acc = (acc * 131 + ord(ch)) % 2147483647
        bits.append(acc % k)
    return sorted(set(bits))


def rappor_encode(values, k=16, h=2, f=0.5, p=0.5, q=0.75, cohorts=1,
                  variant="full", seed=0, client_ids=None):
    r"""Run steps 1-4 for a list of client values.

    Returns the reports, the per-cohort per-bit counts the server sees,
    and the cohort assignment.

    ``client_ids`` labels which client each row came from. Given it, the
    permanent randomized response of step 2 is drawn once per (client,
    value) pair and reused -- the memoization the paper requires -- and
    each client keeps a single cohort across all its rows (Sec. 3.1).
    Left ``None``, every row is treated as a distinct client reporting
    once, which is the same thing when no client repeats.
    """
    var = str(variant).lower()
    if var not in _VARIANTS:
        raise ValueError(
            "rappor_encode: variant must be one of %s, got %r"
            % (", ".join(_VARIANTS), variant))
    vals = list(values)
    n = len(vals)
    if n == 0:
        raise ValueError("rappor_encode: need at least one value")
    k = int(k)
    h = int(h)
    m = int(cohorts)
    if k < 1:
        raise ValueError("rappor_encode: k must be at least 1, got %r" % (k,))
    if m < 1:
        raise ValueError("rappor_encode: cohorts must be at least 1")
    f, p, q = float(f), float(p), float(q)
    _check_params(f, p, q)

    if var == "basic":
        # One-hot over the observed alphabet; no hashing, so h = 1.
        alphabet = sorted(set(str(v) for v in vals))
        k = len(alphabet)
        h = 1
        m = 1
        pos = {v: i for i, v in enumerate(alphabet)}
    else:
        alphabet = None

    if client_ids is None:
        ids = list(range(n))
    else:
        ids = [str(c) for c in client_ids]
        if len(ids) != n:
            raise ValueError(
                "rappor_encode: %d values but %d client_ids"
                % (n, len(ids)))

    rng = np.random.default_rng(seed)
    prr_memo = {}
    cohort_memo = {}
    reports = []
    cohort_of = []
    counts = [[0] * k for _ in range(m)]
    sizes = [0] * m

    for idx in range(n):
        v = vals[idx]
        cid = ids[idx]
        if cid in cohort_memo:
            j = cohort_memo[cid]
        else:
            j = 0 if m == 1 else int(float(rng.uniform()) * m)
            if j >= m:
                j = m - 1
            cohort_memo[cid] = j
        cohort_of.append(j)
        sizes[j] += 1

        B = [0] * k
        if var == "basic":
            B[pos[str(v)]] = 1
        else:
            for b in _bloom(v, k, h, cohort=j):
                B[b] = 1

        # Step 2, PRR -- drawn once per (client, value) and memoized.
        # Re-drawing it per report is what lets an attacker average a
        # client's reports back to B, which is precisely what the
        # permanent response is there to stop.
        memo_key = (cid, str(v))
        if memo_key in prr_memo:
            Bp = list(prr_memo[memo_key])
        else:
            Bp = [0] * k
            for i in range(k):
                u = float(rng.uniform())
                if u < 0.5 * f:
                    Bp[i] = 1
                elif u < f:
                    Bp[i] = 0
                else:
                    Bp[i] = B[i]
            prr_memo[memo_key] = list(Bp)

        # Step 3, IRR -- skipped by the one-time variant.
        if var == "one-time":
            S = Bp
        else:
            S = [0] * k
            for i in range(k):
                thr = q if Bp[i] == 1 else p
                S[i] = 1 if float(rng.uniform()) < thr else 0

        reports.append(S)
        for i in range(k):
            counts[j][i] += S[i]

    return RichResult(payload={
        "estimate": counts,
        "reports": reports,
        "counts": counts,
        "cohort_sizes": sizes,
        "cohort_of": cohort_of,
        "client_ids": list(ids),
        "k": int(k),
        "h": int(h),
        "cohorts": int(m),
        "alphabet": alphabet,
        "variant": var,
        "f": f, "p": p, "q": q,
        "n": int(n),
        "method": "RAPPOR encode (Erlingsson, Pihur & Korolova 2014, Sec. 2)",
    })


def rappor_decode(counts, sizes, f=0.5, p=0.5, q=0.75):
    r"""Section 4's estimator of the true set-bit counts.

    ``t = (c - (p + f q / 2 - f p / 2) N) / ((1 - f)(q - p))``
    """
    f, p, q = float(f), float(p), float(q)
    _check_params(f, p, q)
    denom = (1.0 - f) * (q - p)
    if denom == 0.0:
        raise ValueError(
            "rappor_decode: (1 - f)(q - p) is zero, so the reports carry no "
            "signal and no unbiased estimate exists (f=%r, p=%r, q=%r)"
            % (f, p, q))
    shift = p + 0.5 * f * q - 0.5 * f * p
    rows = [list(r) for r in counts]
    N = [float(v) for v in sizes]
    if len(rows) != len(N):
        raise ValueError(
            "rappor_decode: %d count rows but %d cohort sizes"
            % (len(rows), len(N)))
    est = [[(float(c) - shift * N[j]) / denom for c in rows[j]]
           for j in range(len(rows))]
    return RichResult(payload={
        "estimate": est,
        "t": est,
        "shift": shift,
        "denominator": denom,
        "cohort_sizes": [int(v) for v in N],
        "f": f, "p": p, "q": q,
        "method": "RAPPOR decode (Erlingsson, Pihur & Korolova 2014, Sec. 4)",
    })


def cheatsheet():
    return ("rappor: PRR B'=1/0 w.p. f/2 each else B; IRR P(S=1)=q|p; "
            "eps_inf=2h ln((1-f/2)/(f/2)), eps_1=h log(q*(1-p*)/(p*(1-q*))); "
            "decode t=(c-(p+fq/2-fp/2)N)/((1-f)(q-p)); "
            "variants full/one-time/basic.")


rappor = rappor_encode
