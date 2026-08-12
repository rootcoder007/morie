r"""Circular binary segmentation for DNA copy number.

Olshen, A. B., Venkatraman, E. S., Lucito, R., & Wigler, M. (2004)
"Circular binary segmentation for the analysis of array-based DNA copy
number data", *Biostatistics* 5(4), 557-572.

Copy-number aberrations are discrete gains and losses over *contiguous*
regions, measured noisily, so the task is to split a chromosome into
segments of equal copy number. That is a change-point problem.

Ordinary **binary segmentation** (Sen & Srivastava 1975) tests for one
change at position :math:`i` with

.. math:: Z_i = \Big\{\tfrac{1}{i} + \tfrac{1}{n-i}\Big\}^{-1/2}
          \Big\{\tfrac{S_i}{i} - \tfrac{S_n - S_i}{n - i}\Big\},

:math:`S_i` the partial sum, and recurses on the pieces. Its weakness
is stated plainly in the paper: it "cannot detect a small changed
segment buried in the middle of a large segment", because it looks for
only one change-point at a time -- and a small internal aberration
shifts neither half's mean much.

**Circular binary segmentation** fixes that by splicing the segment's
two ends into a circle and testing whether the arc from :math:`i+1` to
:math:`j` differs from its complement:

.. math:: Z_{ij} = \Big\{\tfrac{1}{j-i} + \tfrac{1}{n-j+i}\Big\}^{-1/2}
          \Big\{\tfrac{S_j - S_i}{j - i}
          - \tfrac{S_n - S_j + S_i}{n - j + i}\Big\},
          \qquad Z_C = \max_{1 \le i < j \le n} |Z_{ij}|.

That statistic covers *both* alternatives at once: a single change when
:math:`j = n`, and the epidemic or square-wave alternative (Levin &
Kline 1985) when :math:`j < n`. Reject when :math:`Z_C` exceeds the
threshold, take the maximising :math:`(i, j)` as the change-points, and
recurse.

Two details from the paper that are easy to skip and are implemented
here:

**The permutation reference distribution.** Rather than lean on
normality, permute the segment, recompute :math:`Z_C^*`, and take the
upper :math:`\alpha` quantile. The paper notes this needs on the order
of 10 000 permutations, and that "considerable computational efficiency
can be achieved by stopping the permutation procedure once the number
of :math:`Z_C^* > Z_C` exceeds :math:`\alpha P`" -- early stopping is
implemented, since it is the difference between usable and not.

**Undoing edge-effect splits.** If the maximising :math:`i` is close to
1 or :math:`j` close to :math:`n`, a ternary split may be an artefact
of a single real change. The paper tests each change-point of a ternary
split for viability as a *binary* split and undoes it if unsupported.

The paper is explicit that it does **not** correct for multiple
testing: :math:`\alpha` is the type I error for a single segment, and
because the procedure recurses "the probability of finding spurious
change-points is a function of the number of true change-points and
could be larger than :math:`\alpha`". That is reported rather than
quietly papered over.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["copynm", "circular_binary_segmentation", "cbs_statistic"]


def _vec(x):
    v = [float(t) for t in np.atleast_1d(np.asarray(x, dtype=float))]
    if not v:
        raise ValueError("copynm: x must be non-empty")
    return v


def cbs_statistic(x):
    r"""Return :math:`Z_C = \max_{i<j} |Z_{ij}|` and its maximiser.

    The statistic of Olshen et al. (2004) section 2. Computed in
    :math:`O(n^2)` from the partial sums, which is what the formula
    costs -- there is no shortcut that preserves the exact maximiser.
    """
    v = _vec(x)
    n = len(v)
    if n < 3:
        raise ValueError("copynm: need at least 3 points to test for a "
                         "change")
    S = [0.0] * (n + 1)
    for t in range(n):
        S[t + 1] = S[t] + v[t]
    Sn = S[n]
    best = -1.0
    bi = bj = 0
    for i in range(0, n):
        for j in range(i + 1, n + 1):
            m = j - i
            k = n - m
            if m == 0 or k == 0:
                continue
            inner = (S[j] - S[i]) / m - (Sn - S[j] + S[i]) / k
            z = abs(inner) / math.sqrt(1.0 / m + 1.0 / k)
            if z > best:
                best, bi, bj = z, i, j
    return best, bi, bj


def _seg_mean(v, a, b):
    return sum(v[a:b]) / float(b - a)


def _binary_supported(v, cut, alpha, perms, rng):
    """Is `cut` a viable BINARY change-point for v? (the undo test)."""
    n = len(v)
    if cut <= 0 or cut >= n:
        return False
    S = sum(v)
    Sc = sum(v[:cut])
    z = abs(Sc / cut - (S - Sc) / (n - cut)) / math.sqrt(
        1.0 / cut + 1.0 / (n - cut))
    # permutation reference for the one-change statistic
    exceed = 0
    limit = alpha * perms
    for p in range(int(perms)):
        w = list(v)
        for t in range(n - 1, 0, -1):
            u = int(rng.random() * (t + 1))
            w[t], w[u] = w[u], w[t]
        Sw = sum(w)
        best = 0.0
        run = 0.0
        for c in range(1, n):
            run += w[c - 1]
            zz = abs(run / c - (Sw - run) / (n - c)) / math.sqrt(
                1.0 / c + 1.0 / (n - c))
            if zz > best:
                best = zz
        if best >= z - 1e-12:
            exceed += 1
            if exceed > limit:
                return False
    return True


def copynm(x, alpha=0.01, permutations=1000, min_width=2, undo_splits=True,
           seed=0, max_depth=50):
    r"""Segment a copy-number profile by circular binary segmentation.

    Parameters
    ----------
    x : array-like
        The log-ratio intensities along one chromosome, in order.
    alpha : float
        Type I error for testing a *single* segment. Note the paper's
        warning: the recursion means the overall false-positive rate is
        not :math:`\alpha`, and no multiple-testing correction is
        applied -- because the true number of change-points is unknown.
    permutations : int
        :math:`P`. The paper suggests on the order of 10 000 for small
        :math:`\alpha`; early stopping makes large :math:`P` cheap when
        the null is clearly true.
    min_width : int
        Minimum segment length to attempt a split on.
    undo_splits : bool
        Apply the paper's edge-effect undo: each change-point of a
        ternary split is tested for viability as a binary split.
    seed : int
        Seed for the permutations.
    max_depth : int
        Recursion guard.

    Returns
    -------
    RichResult
        ``estimate`` / ``segments`` is a list of ``(start, end, mean)``
        with half-open indices; ``changepoints`` the interior
        boundaries; ``n_segments``; ``fitted`` the piecewise-constant
        fit, one value per input point; and ``pvalues`` the permutation
        p-value of each accepted split.

    Examples
    --------
    A small aberration buried in the middle -- the case plain binary
    segmentation is documented to miss::

        x = [0.0]*40 + [1.5]*8 + [0.0]*40
        copynm(x, permutations=200)["changepoints"]   # [40, 48]

    References
    ----------
    Olshen, Venkatraman, Lucito & Wigler (2004) *Biostatistics* 5(4),
    557-572, section 2.
    """
    v = _vec(x)
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("copynm: alpha must lie in (0, 1)")
    permutations = int(permutations)
    if permutations < 1:
        raise ValueError("copynm: permutations must be >= 1")
    min_width = int(min_width)
    if min_width < 2:
        raise ValueError("copynm: min_width must be >= 2")

    rng = np.random.default_rng(seed)
    cuts = set()
    pvals = {}

    def recurse(a, b, depth):
        n = b - a
        if n < max(3, min_width) or depth > max_depth:
            return
        seg = v[a:b]
        # A segment with no variation cannot contain a change-point,
        # and its Z_C is 0. Testing it would compare 0 against a
        # permutation distribution that is also identically 0, and a
        # strict > comparison would then find "no permutation exceeds
        # the observed statistic" and declare a change on constant
        # data. Return early, and count TIES as evidence for the null
        # below, which is the same fix stated generally.
        lo_v, hi_v = min(seg), max(seg)
        if hi_v - lo_v <= 0.0:
            return
        z, i, j = cbs_statistic(seg)
        # permutation reference, with the paper's early stop
        exceed = 0
        limit = alpha * permutations
        used = 0
        for p in range(permutations):
            w = list(seg)
            for t in range(n - 1, 0, -1):
                u = int(rng.random() * (t + 1))
                w[t], w[u] = w[u], w[t]
            zs, _, _ = cbs_statistic(w)
            used += 1
            if zs >= z - 1e-12:
                exceed += 1
                if exceed > limit:
                    break
        pval = (exceed + 1.0) / (used + 1.0)
        if exceed > limit:
            return
        new = []
        if i > 0:
            new.append(a + i)
        if j < n:
            new.append(a + j)
        if not new:
            return
        # Edge-effect undo: a ternary split's change-points are each
        # tested for viability as a binary split.
        if undo_splits and len(new) == 2:
            keep = []
            if _binary_supported(v[a:a + j], i, alpha,
                                 max(50, permutations // 10), rng):
                keep.append(a + i)
            if _binary_supported(v[a + i:b], j - i, alpha,
                                 max(50, permutations // 10), rng):
                keep.append(a + j)
            new = keep
            if not new:
                return
        for c in new:
            cuts.add(c)
            pvals[c] = pval
        bounds = [a] + sorted(new) + [b]
        for t in range(len(bounds) - 1):
            if bounds[t + 1] - bounds[t] < n:
                recurse(bounds[t], bounds[t + 1], depth + 1)

    recurse(0, len(v), 0)

    edges = [0] + sorted(cuts) + [len(v)]
    segs = []
    fitted = [0.0] * len(v)
    for t in range(len(edges) - 1):
        a, b = edges[t], edges[t + 1]
        m = _seg_mean(v, a, b)
        segs.append((a, b, m))
        for q in range(a, b):
            fitted[q] = m
    return RichResult(payload={
        "estimate": segs,
        "segments": segs,
        "changepoints": sorted(cuts),
        "n_segments": len(segs),
        "fitted": fitted,
        "pvalues": dict((c, pvals[c]) for c in sorted(cuts)),
        "alpha": alpha,
        "n": len(v),
        "multiplicity_note": "alpha is the type I error for a SINGLE "
                             "segment; the recursion is not corrected "
                             "for multiple testing (Olshen et al. 2004)",
        "method": "circular binary segmentation (Olshen et al. 2004)",
    })


def cheatsheet():
    return ("copynm: circular binary segmentation (Olshen 2004). Splice "
            "the segment into a circle and test the arc i+1..j against "
            "its complement with Z_ij; Z_C = max|Z_ij| covers BOTH a "
            "single change (j=n) and the epidemic/square-wave "
            "alternative (j<n), which plain binary segmentation misses "
            "for a small aberration in the middle. Permutation "
            "reference with early stopping; ternary splits are undone "
            "if the change-point is not viable as a binary split. "
            "alpha is per-segment and NOT corrected for multiplicity.")


# compact alias per ledger/NAMING.md
circular_binary_segmentation = copynm
