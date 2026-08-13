# morie.fn -- function file (rootcoder007/morie)
"""Private helpers shared by the Rousseeuw high-breakdown function files.

The mirror of R/aaa_helpers_rouss.R.  Every routine performs the same
floating-point operations in the same order as its R counterpart, which
is what lets the parity harness assert agreement at 1e-9.  In
particular the LU factorisation pivots on the same rule in both arms
(largest magnitude, ties broken by the lowest row index), because a
different pivot order gives a different last digit and the two arms
would then disagree on a near-singular scatter matrix.

Nothing here draws a random number.  Where the published algorithms
say "draw random subsets", these helpers enumerate subsets in
lexicographic order instead, so both arms visit the same candidates in
the same sequence and land on the same optimum rather than merely on
an optimum of the same quality.

References
----------
Rousseeuw, P. J., & Van Driessen, K. (1999). "A Fast Algorithm for the
Minimum Covariance Determinant Estimator." *Technometrics* 41(3),
212-223.

None of these PDFs is in the local library; they are cited from
bibliographic details, and the formulas here have not been
re-verified against them.
"""

from __future__ import annotations

import math

__all__: list[str] = []


def osort(v):
    """Indices that sort v ascending, ties broken by the lower index.

    Written out rather than delegated because a language's own sort
    need not be stable in the same way, and the C-step selects "the h
    smallest distances": which of two tied points is taken changes the
    subset and therefore the answer.
    """
    idx = list(range(len(v)))
    for i in range(1, len(idx)):
        j = i
        while j > 0 and (v[idx[j - 1]] > v[idx[j]] or (v[idx[j - 1]] == v[idx[j]] and idx[j - 1] > idx[j])):
            idx[j - 1], idx[j] = idx[j], idx[j - 1]
            j -= 1
    return idx


# Relative tolerance for the pivot-is-zero test.  An EXACT test (bv == 0)
# was the original spelling and it was wrong in a way that mattered: when a
# subset of points lies exactly on a line its covariance is singular, but the
# elimination leaves a pivot of order 1e-13 rather than 0, so the matrix was
# declared non-singular, the "determinant" came out NEGATIVE (-1.6e-12 for a
# covariance matrix, which is impossible), and the Mahalanobis distances
# computed through it were noise.  Downstream, the C-step then cycled with
# period three and its determinant chain INCREASED -- a direct violation of
# the Rousseeuw and Van Driessen theorem the iteration relies on.  The paper
# is explicit that |S| = 0 means the minimum is already attained and the
# iteration must stop, so detecting it correctly is not a nicety.
SINGULAR_TOL = 1e-12


def lufactor(A):
    """LU with partial pivoting.  Returns (LU, piv, sign, singular).

    Singularity is judged RELATIVE to the largest entry of A, not against
    exact zero; see the note on SINGULAR_TOL.
    """
    n = len(A)
    M = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    amax = 0.0
    for i in range(n):
        for j in range(n):
            if abs(M[i][j]) > amax:
                amax = abs(M[i][j])
    thresh = SINGULAR_TOL * amax
    piv = list(range(n))
    sign = 1.0
    singular = False
    for c in range(n):
        best = c
        bv = abs(M[c][c])
        for r in range(c + 1, n):
            if abs(M[r][c]) > bv:
                bv = abs(M[r][c])
                best = r
        if bv <= thresh:
            singular = True
            continue
        if best != c:
            M[c], M[best] = M[best], M[c]
            piv[c], piv[best] = piv[best], piv[c]
            sign = -sign
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r][c] = f
            for j in range(c + 1, n):
                M[r][j] -= f * M[c][j]
    return M, piv, sign, singular


def covdet(S):
    """Determinant of a COVARIANCE matrix, which cannot be negative.

    ludet is a general determinant and may return a small negative
    number for a positive-semidefinite matrix that is singular to
    working precision.  Every objective in this shelf is minimised over
    such determinants, so an unclamped negative rounding artefact wins
    the minimisation outright and the search returns whichever subset
    happened to round furthest below zero.  Clamping at zero makes the
    comparison pick the FIRST exactly-degenerate subset instead, which
    is both correct and identical in the two language arms.
    """
    d = ludet(S)
    return d if d > 0.0 else 0.0


def ludet(A):
    """Determinant of a square matrix by the LU factors."""
    n = len(A)
    if n == 0:
        return 1.0
    M, piv, sign, singular = lufactor(A)
    if singular:
        return 0.0
    d = sign
    for i in range(n):
        d *= M[i][i]
    return d


def lusolve(A, b):
    """Solve A x = b.  Returns None when A is singular."""
    n = len(A)
    M, piv, sign, singular = lufactor(A)
    if singular:
        return None
    y = [0.0] * n
    for i in range(n):
        s = b[piv[i]]
        for j in range(i):
            s -= M[i][j] * y[j]
        y[i] = s
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x


def meancov(X, idx):
    """Mean vector and covariance matrix (divisor |idx| - 1) of a subset."""
    m = len(idx)
    p = len(X[0])
    mu = [0.0] * p
    for i in idx:
        for j in range(p):
            mu[j] += X[i][j]
    mu = [v / float(m) for v in mu]
    S = [[0.0] * p for _ in range(p)]
    for i in idx:
        for a in range(p):
            da = X[i][a] - mu[a]
            for b in range(p):
                S[a][b] += da * (X[i][b] - mu[b])
    den = float(m - 1) if m > 1 else 1.0
    for a in range(p):
        for b in range(p):
            S[a][b] /= den
    return mu, S


def mahal2(X, mu, S):
    """Squared Mahalanobis distances of every row of X.  None if S is singular."""
    n = len(X)
    p = len(mu)
    M, piv, sign, singular = lufactor(S)
    if singular:
        return None
    out = []
    for i in range(n):
        b = [X[i][j] - mu[j] for j in range(p)]
        y = [0.0] * p
        for a in range(p):
            s = b[piv[a]]
            for j in range(a):
                s -= M[a][j] * y[j]
            y[a] = s
        z = [0.0] * p
        for a in range(p - 1, -1, -1):
            s = y[a]
            for j in range(a + 1, p):
                s -= M[a][j] * z[j]
            z[a] = s / M[a][a]
        d = 0.0
        for j in range(p):
            d += b[j] * z[j]
        out.append(d)
    return out


def nchoosek(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def combos(n, k, cap=None):
    """Lexicographic k-subsets of range(n), at most cap of them."""
    out = []
    if k > n or k < 0:
        return out
    c = list(range(k))
    while True:
        out.append(list(c))
        if cap is not None and len(out) >= cap:
            return out
        i = k - 1
        while i >= 0 and c[i] == i + n - k:
            i -= 1
        if i < 0:
            return out
        c[i] += 1
        for j in range(i + 1, k):
            c[j] = c[j - 1] + 1


def cstep(X, idx, h):
    """One C-step of Rousseeuw and Van Driessen (1999).

    Theorem (restated in Hubert, Debruyne and Rousseeuw 2018, arXiv
    1709.07045, "COMPUTATION"): from H1 of size h with mean mu1 and
    covariance S1, taking H2 to be the h observations with the smallest
    distances d(x_i, mu1, S1) gives |S2| <= |S1|, with equality iff
    mu2 = mu1 and S2 = S1.  The determinant therefore never increases,
    which is what makes the iteration terminate.

    Returns (new_idx, det_of_old) or None when S1 is singular, in which
    case the objective is already zero and the subset lies on a
    hyperplane.
    """
    mu, S = meancov(X, idx)
    d0 = covdet(S)
    dd = mahal2(X, mu, S)
    if dd is None:
        return None
    order = osort(dd)
    return sorted(order[:h]), d0


def trimmed_h(n, p):
    """The maximal-breakdown h of Rousseeuw (1984) Remark 1, [n/2] + [(p+1)/2]."""
    return n // 2 + (p + 1) // 2


def mcd_h(n, p):
    """The most robust MCD subset size, [(n + p + 1) / 2]."""
    return (n + p + 1) // 2


def shortest_half(v, h):
    """Shortest window of h points in a sorted univariate sample.

    Rousseeuw (1984) Theorem 2, p. 873: in one dimension the LMS
    location is the midpoint of the shortest half, found as the
    smallest of y_{h:n} - y_{1:n}, ..., y_{n:n} - y_{n-h+1:n}.  The same
    contiguity argument gives the univariate MCD and MVE subsets.

    Returns (start_index_into_sorted, width, sorted_values).
    """
    s = sorted(v)
    n = len(s)
    best = 0
    bw = s[h - 1] - s[0]
    for a in range(1, n - h + 1):
        w = s[a + h - 1] - s[a]
        if w < bw:
            bw = w
            best = a
    return best, bw, s


def combos_stride(n, k, want, max_walk=5000000):
    """`want` k-subsets spread evenly across the whole lexicographic order.

    This exists because of a defect a confusion matrix caught.  The
    published FastMCD draws its elemental subsets AT RANDOM; replacing
    that with "the first `want` subsets in lexicographic order" is not
    a neutral substitution, because the lexicographic prefix is drawn
    almost entirely from the LOWEST indices -- the first 300 of the
    1140 triples of {0..19} never mention an observation past index 9.
    On a fixture whose outliers sat in the upper half, every start was
    therefore seeded inside the clean block's neighbours, the C-steps
    converged on the wrong concentration, and the estimator excluded
    four clean points and kept all four outliers: TP=0 FP=4 TN=12 FN=4,
    a perfectly inverted decision that the determinant alone could not
    reveal.

    Striding keeps the determinism -- both arms visit the same subsets
    in the same order -- while spreading the seeds over the whole index
    range the way random draws would.

    ponytail: walks the enumeration to pick every stride-th subset
    rather than unranking directly, which is O(total) not O(want).
    Fine at the sizes this package enumerates; unrank if that changes.
    """
    total = nchoosek(n, k)
    if total == 0:
        return []
    if total <= want:
        return combos(n, k)
    stride = total // want
    out = []
    c = list(range(k))
    i = 0
    walked = 0
    while True:
        if i % stride == 0:
            out.append(list(c))
            if len(out) >= want:
                return out
        walked += 1
        if walked > max_walk:
            return out
        j = k - 1
        while j >= 0 and c[j] == j + n - k:
            j -= 1
        if j < 0:
            return out
        c[j] += 1
        for m in range(j + 1, k):
            c[m] = c[m - 1] + 1
        i += 1
