# SPDX-License-Identifier: AGPL-3.0-or-later
"""Matrix profile (self-join) for discord and motif detection."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["matrxP", "matrix_profile"]


def matrxP(x, window):
    """
    Self-join matrix profile.

    Yeh et al. (2016): the matrix profile P (Definition 7) annotates a
    series T with, for each subsequence T_{i,m} (Definition 2), the
    z-normalised Euclidean distance to its nearest neighbour in the
    all-subsequences set (Definition 4), excluding trivial matches via
    an exclusion zone of m/2 positions before and after i (their
    Sec. II.A discussion of Figure 2 and Definition 8). The matrix
    profile index I (Definition 9) records where that neighbour is.
    Distances use the z-normalised Euclidean distance computed from
    dot products (their Table II / MASS discussion):

      D[i] = sqrt( 2 m (1 - (QT[i] - m mu_Q M_T[i]) /
                          (m sigma_Q Sigma_T[i])) ),

    with mu/M the subsequence means and sigma/Sigma the (population)
    standard deviations. The highest point of P is the discord and
    the lowest pair is the best motif (their Sec. II, properties of
    the profile).

    Parameters
    ----------
    x : array-like
        Series T.
    window : int
        Subsequence length m.

    Returns
    -------
    result : RichResult
        Keys: profile, index (1-based neighbour locations), discord
        (1-based location of the profile maximum), discord_distance,
        motif (1-based pair), motif_distance.

    References
    ----------
    Yeh, C.-C. M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Trinh,
    H. A., Silva, D. F., Mueen, A. and Keogh, E. (2016), "Matrix
    profile I: all pairs similarity joins for time series", Proc. 16th
    IEEE International Conference on Data Mining (ICDM 2016),
    pp. 1317-1322. Definitions 1-9, Table II (z-normalised distance
    formula), STAMP (Table III).
    Source PDF: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/
    yeh-etal-2016-matrix-profile-i-icdm.pdf
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float))
    xs = [float(v) for v in xv]
    nlen = len(xs)
    m = int(window)
    if m < 2 or m > nlen // 2:
        raise ValueError("need 2 <= window <= len(x)/2")
    nsub = nlen - m + 1
    # cached cumulative sums for subsequence means / sds (their
    # Sec. III.A note on computing MeanStd in O(1) per subsequence)
    cs = [0.0] * (nlen + 1)
    css = [0.0] * (nlen + 1)
    for i in range(nlen):
        cs[i + 1] = cs[i] + xs[i]
        css[i + 1] = css[i] + xs[i] * xs[i]
    mu = [0.0] * nsub
    sd = [0.0] * nsub
    for i in range(nsub):
        s = cs[i + m] - cs[i]
        ss = css[i + m] - css[i]
        mu[i] = s / m
        v = ss / m - mu[i] * mu[i]
        sd[i] = math.sqrt(v) if v > 0.0 else 0.0
    excl = m // 2
    P = [math.inf] * nsub
    I = [-1] * nsub
    for i in range(nsub):
        for j in range(i + 1, nsub):
            if j - i <= excl:
                continue
            qt = 0.0
            for t in range(m):
                qt += xs[i + t] * xs[j + t]
            if sd[i] <= 0.0 or sd[j] <= 0.0:
                # constant subsequence: fall back to distance between
                # z-normalised forms (all-zero vectors)
                d = 0.0 if (sd[i] <= 0.0 and sd[j] <= 0.0) else math.sqrt(2.0 * m)
            else:
                arg = 1.0 - (qt - m * mu[i] * mu[j]) / (m * sd[i] * sd[j])
                if arg < 0.0:
                    arg = 0.0
                d = math.sqrt(2.0 * m * arg)
            if d < P[i]:
                P[i] = d
                I[i] = j
            if d < P[j]:
                P[j] = d
                I[j] = i
    ib = 0
    iw = 0
    for i in range(1, nsub):
        if P[i] > P[ib]:
            ib = i
        if P[i] < P[iw]:
            iw = i
    return RichResult(payload={
        "profile": [float(v) for v in P],
        "index": [i + 1 for i in I],
        "discord": ib + 1,
        "discord_distance": float(P[ib]),
        "motif": [iw + 1, I[iw] + 1],
        "motif_distance": float(P[iw]),
        "window": m,
        "estimate": ib + 1,
        "n": nlen,
        "method": "Matrix profile self-join (Yeh et al. 2016)",
    })


def matrix_profile(x, window):
    """Alias for matrxP (original stub export name)."""
    return matrxP(x, window)


def cheatsheet():
    return "matrxP(x, m) -> matrix profile + index; max = discord, min = motif"

# public names resolved by fn/_lazy_map.json
matrixprofile = matrxP
