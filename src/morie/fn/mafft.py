# morie.fn -- function file (rootcoder007/morie)
r"""MAFFT: multiple sequence alignment through the fast Fourier transform.

Katoh, K., Misawa, K., Kuma, K., & Miyata, T. (2002) "MAFFT: a novel
method for rapid multiple sequence alignment based on fast Fourier
transform", *Nucleic Acids Research* 30(14), 3059-3066.
doi:10.1093/nar/gkf436

Katoh, K., Kuma, K., Toh, H., & Miyata, T. (2005) "MAFFT version 5:
improvement in accuracy of multiple sequence alignment", *NAR* 33(2),
511-518. doi:10.1093/nar/gki198

Katoh, K., & Standley, D. M. (2013) "MAFFT Multiple Sequence Alignment
Software Version 7", *MBE* 30(4), 772-780. doi:10.1093/molbev/mst010

The 2013 paper is the options table; the algorithm below is the 2002
paper's, which is where the FFT, the scoring system and the three named
methods are actually defined.

**Why a Fourier transform.** Substitution rates depend strongly on the
physico-chemical difference between the two residues, so each amino acid
is given Grantham's volume :math:`v(a)` and polarity :math:`p(a)`, in
normalised form :math:`\hat{v}(a) = [v(a) - \bar{v}]/\sigma_v` and
likewise for polarity, and a sequence becomes a sequence of 2-vectors.
The similarity of two sequences offset by :math:`k` sites is then a
correlation,

.. math::

   c(k) = c_v(k) + c_p(k), \qquad
   c_v(k) = \sum_n \hat{v}_1(n)\, \hat{v}_2(n + k),

which is :math:`O(N^2)` written that way but is a transform pair,
:math:`c_v(k) \Leftrightarrow V_1^{*}(m) \cdot V_2(m)`, so the FFT gets
it in :math:`O(N \log N)`. Nucleotides use 4-vectors of A/T/G/C
frequency and :math:`c(k) = c_A + c_T + c_G + c_C`.

A peak in :math:`c(k)` gives the *lag* of a homologous region but not
where it sits, so a sliding window of **30 sites** is run over the
highest **20** peaks; a window scoring above **0.7 per site** is a
homologous segment, consecutive segments are merged, and any merged run
over **150 sites** is cut into 150-site pieces. Those segments are then
arranged consistently by a DP over a segment matrix :math:`S_{ij}`, and
the residue-level DP is split into sub-matrices at the segment centres
(Figure 2B) -- which is where the speed comes from, since the shaded
corners never get computed.

**The scoring system is deliberately not all-positive.** Equation 7
rescales any raw matrix,

.. math::

   \hat{M}_{ab} = \frac{M_{ab} - \text{average2}}
                       {\text{average1} - \text{average2}} + S_a,

with :math:`\text{average1} = \sum_a f_a M_{aa}` and
:math:`\text{average2} = \sum_{a,b} f_a f_b M_{ab}`. Under it the score
per site is exactly :math:`S_a` between two random sequences and exactly
:math:`1 + S_a` between two identical ones, so a gap is scored roughly
like random sequence when :math:`S_a` is small. The paper's controls are
kept as routes: ``matrix="all_positive"`` reproduces NW-AP-2 by choosing
the :math:`S_a` that lifts every entry positive, and ``method="NW-NS-2"``
skips the FFT entirely.

**Gap penalty.** A gap opened where the group already has one should not
be paid for twice, so

.. math::

   G_1(i, x) = S_{op}\Bigl\{1 - \tfrac{1}{2}
               \bigl[g^{start}_1(x) + g^{end}_1(i)\bigr]\Bigr\},

where :math:`g^{start}_1(x) = \sum_m w_m a_m(x) z_m(x+1)` counts (by
weight) the gaps starting just after :math:`x` and
:math:`g^{end}_1(i) = \sum_m w_m z_m(i-1) a_m(i)` those ending just
before :math:`i`, with :math:`z_m = 1` on a gap and :math:`a_m = 1`
otherwise. When every sequence in the group already carries that gap
both terms are 1 and the penalty is exactly zero.

**The three methods** (all implemented, ``method=``):

``"FFT-NS-1"``
    guide tree from the 6-tuple distance
    :math:`D_{ij} = 1 - T_{ij}/\min(T_{ii}, T_{jj})` over six
    physico-chemical residue groups, UPGMA, then one progressive pass.
``"FFT-NS-2"`` (default)
    the tree is rebuilt from the FFT-NS-1 alignment and the progressive
    pass is run again -- "more reliable alignments ... on the basis of
    more reliable guide trees".
``"FFT-NS-i"``
    FFT-NS-2 followed by iterative refinement: split the alignment in
    two along an edge of the guide tree, realign the halves, keep the
    result only if the weighted sum-of-pairs score improves, and repeat
    until nothing improves.

Sequences are plain strings; groups are lists of equal-length gapped
strings. Sizes are the ones an anchor can check rather than the
thousands of sequences the 2013 paper benchmarks.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "mafft_alignment",
    "mafftalignment",
    "residue_vectors",
    "correlation",
    "find_homologous_segments",
    "arrange_segments",
    "normalized_similarity_matrix",
    "group_align",
    "sixtuple_distance",
    "guide_tree",
    "progressive_align",
    "iterative_refine",
    "wsp_score",
    "GRANTHAM_POLARITY",
    "GRANTHAM_VOLUME",
]

_AA = "ARNDCQEGHILKMFPSTWYV"
_NT = "ACGT"

#: Grantham's polarity, as MAFFT itself carries it (``core/miyata.h``).
GRANTHAM_POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
    "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}

#: Grantham's volume, likewise.
GRANTHAM_VOLUME = {
    "A": 31.0, "R": 124.0, "N": 56.0, "D": 54.0, "C": 55.0,
    "Q": 85.0, "E": 83.0, "G": 3.0, "H": 96.0, "I": 111.0,
    "L": 111.0, "K": 119.0, "M": 105.0, "F": 132.0, "P": 32.5,
    "S": 32.0, "T": 61.0, "W": 170.0, "Y": 136.0, "V": 84.0,
}

_METHODS = ("FFT-NS-1", "FFT-NS-2", "FFT-NS-i", "NW-NS-1", "NW-NS-2")
_MATRICES = ("normalized", "all_positive")

# The six physico-chemical groups used for the 6-tuple distance.
_SIX_GROUPS = ("AGPST", "C", "DENQ", "FWY", "HKR", "ILMV")


def _norm(vals):
    n = float(len(vals))
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / n)
    if sd <= 0:
        raise ValueError("mafft: a property with no variation cannot be "
                         "normalised")
    return mu, sd


_PMU, _PSD = _norm([GRANTHAM_POLARITY[a] for a in _AA])
_VMU, _VSD = _norm([GRANTHAM_VOLUME[a] for a in _AA])

#: normalised Grantham vectors, one per residue
_VHAT = dict((a, (GRANTHAM_VOLUME[a] - _VMU) / _VSD) for a in _AA)
_PHAT = dict((a, (GRANTHAM_POLARITY[a] - _PMU) / _PSD) for a in _AA)


# ------------------------------------------------------------ sequences

def _clean(seqs, seq_type=None):
    out = []
    for s in seqs:
        t = str(s).upper()
        if not t:
            raise ValueError("mafft: an empty sequence was given")
        out.append(t)
    if seq_type is None:
        letters = set("".join(out)) - set("-.")
        seq_type = "nt" if letters and letters <= set(_NT + "UN") else "aa"
    if seq_type not in ("aa", "nt"):
        raise ValueError("mafft: seq_type must be 'aa' or 'nt'")
    return out, seq_type


def residue_vectors(group, weights=None, seq_type="aa"):
    r"""The vector sequences of Equations 2 and 6, for a group.

    A group of one is the plain sequence of the paper; a larger group
    uses the weighted linear combination
    :math:`\hat{v}_{group}(n) = \sum_i w_i \hat{v}_i(n)`. Gaps
    contribute nothing. Returns a list of component sequences: two for
    amino acids (volume, polarity), four for nucleotides.
    """
    rows = [str(s).upper() for s in group]
    if not rows:
        raise ValueError("mafft: an empty group has no vectors")
    L = len(rows[0])
    for r in rows:
        if len(r) != L:
            raise ValueError("mafft: sequences in a group must be aligned "
                             "to the same length")
    if weights is None:
        weights = [1.0 / len(rows)] * len(rows)
    weights = [float(w) for w in weights]
    if len(weights) != len(rows):
        raise ValueError("mafft: one weight per sequence is required")
    if seq_type == "nt":
        comps = []
        for base in _NT:
            comps.append([sum(w for w, r in zip(weights, rows)
                              if r[n] == base) for n in range(L)])
        return comps
    vol = [sum(w * _VHAT.get(r[n], 0.0) for w, r in zip(weights, rows))
           for n in range(L)]
    pol = [sum(w * _PHAT.get(r[n], 0.0) for w, r in zip(weights, rows))
           for n in range(L)]
    return [vol, pol]


# ---------------------------------------------------------- correlation

def _xcorr_fft(a, b):
    """``sum_n a(n) b(n+k)`` for every lag, via one transform pair."""
    n, m = len(a), len(b)
    size = 1
    while size < n + m:
        size *= 2
    fa = np.fft.fft(list(a) + [0.0] * (size - n))
    fb = np.fft.fft(list(b) + [0.0] * (size - m))
    prod = [complex(x).conjugate() * complex(y) for x, y in zip(fa, fb)]
    back = np.fft.ifft(prod)
    return [complex(v).real for v in back], size


def _xcorr_direct(a, b, size):
    """Equation 2 written out, over the same circular index the FFT uses.

    The transform wraps, and the wrap is not an artefact: index
    ``(n + k) mod size`` for ``k`` in the upper half is exactly the
    negative lag ``k - size``. Summing without the wrap would compute a
    different quantity, and comparing the two would prove nothing.
    """
    out = [0.0] * size
    n, m = len(a), len(b)
    for k in range(size):
        tot = 0.0
        for i in range(n):
            j = (i + k) % size
            if j < m:
                tot += a[i] * b[j]
        out[k] = tot
    return out


def correlation(group1, group2, weights1=None, weights2=None,
                seq_type="aa", method="fft"):
    r"""The correlation :math:`c(k)` of Equation 1, for every lag.

    ``method="fft"`` uses the transform pair of Equation 5;
    ``method="direct"`` evaluates Equation 2 as written, which is the
    :math:`O(N^2)` definition the FFT is supposed to reproduce exactly.
    Returns ``(lags, c)`` with positive lags first and then the negative
    ones, as the transform produces them.
    """
    if method not in ("fft", "direct"):
        raise ValueError("mafft: method must be 'fft' or 'direct'")
    c1 = residue_vectors(group1, weights1, seq_type)
    c2 = residue_vectors(group2, weights2, seq_type)
    total, size = None, None
    for a, b in zip(c1, c2):
        if method == "fft":
            part, size = _xcorr_fft(a, b)
        else:
            if size is None:
                size = 1
                while size < len(a) + len(b):
                    size *= 2
            part = _xcorr_direct(a, b, size)
        total = part if total is None else [x + y for x, y in
                                            zip(total, part)]
    half = size // 2
    lags = list(range(half)) + list(range(-half, 0))
    return lags, total


def _peaks(lags, c, n_peaks):
    order = sorted(range(len(c)), key=lambda i: -c[i])
    return [lags[i] for i in order[:int(n_peaks)]]


# ------------------------------------------------------------- scoring

def _default_raw_matrix(seq_type):
    """A raw matrix derived from the same Grantham vectors as the FFT.

    The paper's default is the 200-PAM JTT log-odds matrix; that matrix
    is not reproduced here, so the default is built from the physico-
    chemical distance the method already relies on,
    :math:`M_{ab} = -[(\\hat{v}_a - \\hat{v}_b)^2 +
    (\\hat{p}_a - \\hat{p}_b)^2]`. Pass ``raw_matrix`` and ``freqs`` to
    use JTT, BLOSUM or anything else -- Equation 7 rescales whatever it
    is given.
    """
    if seq_type == "nt":
        return dict(((a, b), 1.0 if a == b else -1.0)
                    for a in _NT for b in _NT)
    M = {}
    for a in _AA:
        for b in _AA:
            M[(a, b)] = -((_VHAT[a] - _VHAT[b]) ** 2 +
                          (_PHAT[a] - _PHAT[b]) ** 2)
    return M


def normalized_similarity_matrix(raw_matrix=None, freqs=None, s_a=0.06,
                                 seq_type="aa", mode="normalized"):
    r"""Equation 7.

    ``mode="all_positive"`` is the paper's NW-AP-2 control: the raw
    matrix made positive by subtracting its smallest element, which is
    the same as choosing the :math:`S_a` that just lifts every entry
    above zero.
    """
    if mode not in _MATRICES:
        raise ValueError("mafft: mode must be one of %s" % (_MATRICES,))
    alpha = _NT if seq_type == "nt" else _AA
    M = dict(_default_raw_matrix(seq_type)) if raw_matrix is None \
        else dict(raw_matrix)
    if freqs is None:
        freqs = dict((a, 1.0 / len(alpha)) for a in alpha)
    freqs = dict(freqs)
    tot = sum(freqs.get(a, 0.0) for a in alpha)
    if tot <= 0:
        raise ValueError("mafft: frequencies must be positive")
    freqs = dict((a, freqs.get(a, 0.0) / tot) for a in alpha)
    for a in alpha:
        for b in alpha:
            if (a, b) not in M:
                raise ValueError("mafft: raw_matrix is missing (%s, %s)"
                                 % (a, b))
    avg1 = sum(freqs[a] * M[(a, a)] for a in alpha)
    avg2 = sum(freqs[a] * freqs[b] * M[(a, b)]
               for a in alpha for b in alpha)
    if abs(avg1 - avg2) < 1e-15:
        raise ValueError("mafft: raw_matrix has no signal (average1 equals "
                         "average2)")
    base = dict(((a, b), (M[(a, b)] - avg2) / (avg1 - avg2))
                for a in alpha for b in alpha)
    if mode == "all_positive":
        s_a = -min(base.values())
    out = dict((k, v + s_a) for k, v in base.items())
    return {"matrix": out, "s_a": float(s_a), "alphabet": alpha,
            "average1": avg1, "average2": avg2, "freqs": freqs,
            "mode": mode}


def _site_score(M, ga, gb, wa, wb, i, j):
    """``H(i, j)``: the weighted average matrix score of two columns."""
    tot = 0.0
    for wn, sn in zip(wa, ga):
        a = sn[i]
        if a == "-":
            continue
        for wm, sm in zip(wb, gb):
            b = sm[j]
            if b == "-":
                continue
            tot += wn * wm * M.get((a, b), 0.0)
    return tot


def _gap_profiles(group, weights):
    """``g_start`` and ``g_end`` of the gap penalty, per position."""
    L = len(group[0])
    gs, ge = [0.0] * (L + 1), [0.0] * (L + 1)
    for w, s in zip(weights, group):
        z = [1.0 if ch == "-" else 0.0 for ch in s]
        a = [1.0 - v for v in z]
        for x in range(L):
            nxt = z[x + 1] if x + 1 < L else 0.0
            gs[x] += w * a[x] * nxt          # a gap starts just after x
            prv = z[x - 1] if x - 1 >= 0 else 0.0
            ge[x] += w * prv * a[x]          # a gap ended just before x
    return gs, ge


# ------------------------------------------------------- group alignment

def group_align(group1, group2, scoring, weights1=None, weights2=None,
                s_op=2.4, anchors=None):
    """Align two groups by the paper's NW recursion and gap penalty.

    ``anchors`` is a list of ``(i, j)`` residue pairs that the alignment
    must pass through -- the segment centres of Figure 2B. With them the
    DP runs on the sub-matrices between consecutive anchors instead of
    the whole rectangle.
    """
    g1 = [str(s).upper() for s in group1]
    g2 = [str(s).upper() for s in group2]
    if not g1 or not g2:
        raise ValueError("mafft: both groups must be non-empty")
    for g in (g1, g2):
        if len(set(len(s) for s in g)) != 1:
            raise ValueError("mafft: a group must be aligned to one length")
    w1 = [1.0 / len(g1)] * len(g1) if weights1 is None else list(weights1)
    w2 = [1.0 / len(g2)] * len(g2) if weights2 is None else list(weights2)
    if len(w1) != len(g1) or len(w2) != len(g2):
        raise ValueError("mafft: one weight per sequence is required")
    M = scoring["matrix"] if isinstance(scoring, dict) else scoring
    if anchors:
        n, m = len(g1[0]), len(g2[0])
        given = sorted(set(tuple(a) for a in anchors))
        # Anchors are only usable if they can all lie on one path: sorted
        # by the first coordinate, the second must not go backwards.
        # Crossing anchors describe two incompatible alignments and are a
        # caller error, not something to quietly sort away.
        for u, v in zip(given, given[1:]):
            if v[1] < u[1]:
                raise ValueError("mafft: anchors cross and cannot lie on "
                                 "one alignment path")
        for a, b in given:
            if not (0 <= a <= n and 0 <= b <= m):
                raise ValueError("mafft: an anchor is outside the groups")
        pts = sorted(set([(0, 0)] + given + [(n, m)]))
        out1, out2 = [""] * len(g1), [""] * len(g2)
        prev = pts[0]
        for pt in pts[1:]:
            a1 = [s[prev[0]:pt[0]] for s in g1]
            a2 = [s[prev[1]:pt[1]] for s in g2]
            if not a1[0] and not a2[0]:
                prev = pt
                continue
            p1, p2 = _nw(a1, a2, M, w1, w2, s_op)
            out1 = [o + p for o, p in zip(out1, p1)]
            out2 = [o + p for o, p in zip(out2, p2)]
            prev = pt
        return out1, out2
    return _nw(g1, g2, M, w1, w2, s_op)


def _nw(g1, g2, M, w1, w2, s_op):
    n = len(g1[0]) if g1[0] else 0
    m = len(g2[0]) if g2[0] else 0
    if n == 0:
        return ["-" * m for _ in g1], [s for s in g2]
    if m == 0:
        return [s for s in g1], ["-" * n for _ in g2]
    gs1, ge1 = _gap_profiles(g1, w1)
    gs2, ge2 = _gap_profiles(g2, w2)
    neg = float("-inf")
    P = [[neg] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    P[0][0] = 0.0
    for i in range(1, n + 1):
        P[i][0] = -s_op * (1.0 - (gs1[0] + ge1[i - 1]) / 2.0)
        back[i][0] = ("I", 0, 0)
    for j in range(1, m + 1):
        P[0][j] = -s_op * (1.0 - (gs2[0] + ge2[j - 1]) / 2.0)
        back[0][j] = ("D", 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            h = _site_score(M, g1, g2, w1, w2, i - 1, j - 1)
            best = (P[i - 1][j - 1], ("M", i - 1, j - 1))
            for x in range(0, i):
                if P[x][j - 1] == neg:
                    continue
                pen = s_op * (1.0 - (gs1[x] + ge1[i - 1]) / 2.0)
                v = P[x][j - 1] - pen
                if v > best[0]:
                    best = (v, ("I", x, j - 1))
            for y in range(0, j):
                if P[i - 1][y] == neg:
                    continue
                pen = s_op * (1.0 - (gs2[y] + ge2[j - 1]) / 2.0)
                v = P[i - 1][y] - pen
                if v > best[0]:
                    best = (v, ("D", i - 1, y))
            P[i][j] = h + best[0]
            back[i][j] = best[1]
    # Trace back. Every branch of the recursion adds H(i, j), so (i-1,
    # j-1) is a matched column whichever branch was taken; a gap branch
    # adds a run of gapped columns BEFORE that match. Emitting only the
    # run silently drops a residue.
    cols = []
    i, j = n, m
    while i > 0 and j > 0:
        kind, pi, pj = back[i][j]
        cols.append((i - 1, j - 1))
        if kind == "I":
            for t in range(i - 2, pi - 1, -1):
                cols.append((t, None))
        elif kind == "D":
            for t in range(j - 2, pj - 1, -1):
                cols.append((None, t))
        i, j = pi, pj
    for t in range(i - 1, -1, -1):
        cols.append((t, None))
    for t in range(j - 1, -1, -1):
        cols.append((None, t))
    cols.reverse()
    out1 = ["".join(s[c1] if c1 is not None else "-" for c1, _ in cols)
            for s in g1]
    out2 = ["".join(s[c2] if c2 is not None else "-" for _, c2 in cols)
            for s in g2]
    return out1, out2


# --------------------------------------------------------- FFT anchoring

def find_homologous_segments(group1, group2, scoring, weights1=None,
                             weights2=None, seq_type="aa", window=30,
                             n_peaks=20, threshold=0.7, max_len=150,
                             corr_method="fft"):
    """The sliding-window step: peaks of ``c(k)`` become segments.

    Each of the ``n_peaks`` highest peaks is walked with a window of
    ``window`` sites; a window scoring above ``threshold`` per site is a
    homologous segment, runs of them are merged, and merged runs longer
    than ``max_len`` are cut into ``max_len`` pieces.
    """
    if window < 1 or n_peaks < 1 or max_len < 1:
        raise ValueError("mafft: window, n_peaks and max_len must be "
                         "positive")
    M = scoring["matrix"] if isinstance(scoring, dict) else scoring
    g1 = [str(s).upper() for s in group1]
    g2 = [str(s).upper() for s in group2]
    w1 = [1.0 / len(g1)] * len(g1) if weights1 is None else list(weights1)
    w2 = [1.0 / len(g2)] * len(g2) if weights2 is None else list(weights2)
    n, m = len(g1[0]), len(g2[0])
    lags, c = correlation(g1, g2, w1, w2, seq_type, corr_method)
    segs = []
    for k in _peaks(lags, c, n_peaks):
        lo = max(0, -k)
        hi = min(n, m - k)
        if hi - lo < window:
            continue
        run = None
        for start in range(lo, hi - window + 1):
            score = sum(_site_score(M, g1, g2, w1, w2, start + t,
                                    start + t + k)
                        for t in range(window)) / float(window)
            if score > threshold:
                run = (run[0], start + window, run[2] + [score]) if run \
                    else (start, start + window, [score])
            elif run:
                segs.append((run[0], run[0] + k, run[1] - run[0],
                             sum(run[2]) / len(run[2]), k))
                run = None
        if run:
            segs.append((run[0], run[0] + k, run[1] - run[0],
                         sum(run[2]) / len(run[2]), k))
    # cut anything longer than max_len
    cut = []
    for s1, s2, ln, sc, k in segs:
        while ln > max_len:
            cut.append((s1, s2, max_len, sc, k))
            s1 += max_len
            s2 += max_len
            ln -= max_len
        if ln > 0:
            cut.append((s1, s2, ln, sc, k))
    cut.sort()
    return cut


def arrange_segments(segments):
    """The segment-level DP of Figure 2A.

    Builds :math:`S_{ij}` over the segments and takes the highest-scoring
    consistent (strictly increasing in both sequences) chain, which is
    the optimal arrangement. Returns the chosen segments in order.
    """
    segs = sorted(segments)
    n = len(segs)
    if n == 0:
        return []
    best = [s[3] * s[2] for s in segs]
    prev = [None] * n
    for i in range(n):
        for j in range(i):
            a, b = segs[j], segs[i]
            if a[0] + a[2] <= b[0] and a[1] + a[2] <= b[1]:
                v = best[j] + segs[i][3] * segs[i][2]
                if v > best[i]:
                    best[i] = v
                    prev[i] = j
    end = max(range(n), key=lambda i: best[i])
    chain = []
    while end is not None:
        chain.append(segs[end])
        end = prev[end]
    chain.reverse()
    return chain


def _anchors_from(chain):
    """Segment centres, which is where the homology matrix is divided."""
    return [(s[0] + s[2] // 2, s[1] + s[2] // 2) for s in chain
            if s[0] >= 0 and s[1] >= 0]


# ---------------------------------------------------- trees and progress

def sixtuple_distance(seqs):
    r""":math:`D_{ij} = 1 - T_{ij}/\min(T_{ii}, T_{jj})`.

    :math:`T_{ij}` counts 6-tuples shared by the two sequences after the
    20 residues are collapsed into six physico-chemical groups.
    """
    coded = []
    for s in seqs:
        t = ""
        for ch in str(s).upper():
            if ch == "-":
                continue
            for gi, grp in enumerate(_SIX_GROUPS):
                if ch in grp:
                    t += chr(ord("a") + gi)
                    break
            else:
                t += "z"
        coded.append(t)

    def tuples(t):
        d = {}
        for i in range(len(t) - 5):
            key = t[i:i + 6]
            d[key] = d.get(key, 0) + 1
        return d

    tabs = [tuples(t) for t in coded]

    def shared(a, b):
        return sum(min(v, b.get(k, 0)) for k, v in a.items())

    n = len(seqs)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = min(shared(tabs[i], tabs[i]), shared(tabs[j], tabs[j]))
            t = shared(tabs[i], tabs[j])
            D[i][j] = 1.0 - (t / float(denom) if denom else 0.0)
    return D


def guide_tree(D):
    """UPGMA over a distance matrix; returns the merge order."""
    n = len(D)
    if n < 2:
        raise ValueError("mafft: a guide tree needs at least two sequences")
    clusters = dict((i, [i]) for i in range(n))
    dist = dict(((i, j), D[i][j]) for i in range(n) for j in range(n)
                if i != j)
    merges = []
    nxt = n
    active = list(range(n))
    while len(active) > 1:
        best = min(((dist[(i, j)], i, j) for k, i in enumerate(active)
                    for j in active[k + 1:]), key=lambda t: (t[0], t[1], t[2]))
        _, i, j = best
        members = clusters[i] + clusters[j]
        merges.append((i, j, nxt, list(members)))
        for k in active:
            if k in (i, j):
                continue
            ni, nj = len(clusters[i]), len(clusters[j])
            d = (ni * dist[(i, k)] + nj * dist[(j, k)]) / float(ni + nj)
            dist[(nxt, k)] = d
            dist[(k, nxt)] = d
        clusters[nxt] = members
        active = [k for k in active if k not in (i, j)] + [nxt]
        nxt += 1
    return merges


def _weights(k):
    return [1.0 / k] * k


def progressive_align(seqs, scoring, tree=None, seq_type="aa", s_op=2.4,
                      use_fft=True, **kw):
    """One progressive pass along the guide tree."""
    seqs = [str(s).upper() for s in seqs]
    if len(seqs) < 2:
        raise ValueError("mafft: at least two sequences are needed")
    if tree is None:
        tree = guide_tree(sixtuple_distance(seqs))
    profiles = dict((i, [seqs[i]]) for i in range(len(seqs)))
    members = dict((i, [i]) for i in range(len(seqs)))
    for i, j, new, _ in tree:
        g1, g2 = profiles[i], profiles[j]
        anchors = None
        if use_fft:
            segs = find_homologous_segments(g1, g2, scoring,
                                            _weights(len(g1)),
                                            _weights(len(g2)), seq_type,
                                            **kw)
            anchors = _anchors_from(arrange_segments(segs)) or None
        a1, a2 = group_align(g1, g2, scoring, _weights(len(g1)),
                             _weights(len(g2)), s_op, anchors)
        profiles[new] = a1 + a2
        members[new] = members[i] + members[j]
        del profiles[i], profiles[j]
    root = list(profiles)[0]
    order = members[root]
    out = [None] * len(seqs)
    for pos, idx in enumerate(order):
        out[idx] = profiles[root][pos]
    return out


def wsp_score(alignment, scoring, s_op=2.4, weights=None):
    """Weighted sum-of-pairs score of an alignment."""
    aln = [str(s).upper() for s in alignment]
    if len(set(len(s) for s in aln)) != 1:
        raise ValueError("mafft: an alignment must be rectangular")
    M = scoring["matrix"] if isinstance(scoring, dict) else scoring
    k = len(aln)
    w = _weights(k) if weights is None else list(weights)
    total = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            pair = w[i] * w[j]
            opened = False
            for a, b in zip(aln[i], aln[j]):
                if a == "-" or b == "-":
                    if not opened:
                        total -= pair * s_op
                        opened = True
                else:
                    opened = False
                    total += pair * M.get((a, b), 0.0)
    return total


def iterative_refine(alignment, scoring, tree=None, s_op=2.4,
                     max_iterate=16, seq_type="aa", use_fft=True, **kw):
    """FFT-NS-i: split along a tree edge, realign, keep if WSP improves."""
    aln = [str(s).upper() for s in alignment]
    if max_iterate < 1:
        raise ValueError("mafft: max_iterate must be at least 1")
    best = wsp_score(aln, scoring, s_op)
    if tree is None:
        tree = guide_tree(sixtuple_distance([s.replace("-", "")
                                             for s in aln]))
    groups = []
    for _, _, _, members in tree[:-1]:
        rest = [i for i in range(len(aln)) if i not in members]
        if members and rest:
            groups.append((list(members), rest))
    rounds = 0
    for _ in range(int(max_iterate)):
        improved = False
        for members, rest in groups:
            g1 = _degap([aln[i] for i in members])
            g2 = _degap([aln[i] for i in rest])
            anchors = None
            if use_fft:
                segs = find_homologous_segments(g1, g2, scoring,
                                                _weights(len(g1)),
                                                _weights(len(g2)),
                                                seq_type, **kw)
                anchors = _anchors_from(arrange_segments(segs)) or None
            a1, a2 = group_align(g1, g2, scoring, _weights(len(g1)),
                                 _weights(len(g2)), s_op, anchors)
            cand = [None] * len(aln)
            for pos, idx in enumerate(members):
                cand[idx] = a1[pos]
            for pos, idx in enumerate(rest):
                cand[idx] = a2[pos]
            sc = wsp_score(cand, scoring, s_op)
            if sc > best + 1e-12:
                aln, best, improved = cand, sc, True
        rounds += 1
        if not improved:
            break
    return aln, best, rounds


def _degap(group):
    """Drop columns that are all gaps, so a sub-group stays rectangular."""
    if not group:
        return group
    L = len(group[0])
    keep = [i for i in range(L) if any(s[i] != "-" for s in group)]
    return ["".join(s[i] for i in keep) for s in group]


# -------------------------------------------------------------- driver

def mafft_alignment(sequences, method="FFT-NS-2", seq_type=None,
                    raw_matrix=None, freqs=None, s_a=0.06, s_op=2.4,
                    matrix="normalized", window=30, n_peaks=20,
                    threshold=0.7, max_len=150, max_iterate=16):
    """Align ``sequences`` by one of the paper's named methods."""
    if method not in _METHODS:
        raise ValueError("mafft: method must be one of %s" % (_METHODS,))
    seqs, kind = _clean(sequences, seq_type)
    if len(seqs) < 2:
        raise ValueError("mafft: at least two sequences are needed")
    sc = normalized_similarity_matrix(raw_matrix, freqs, s_a, kind, matrix)
    use_fft = method.startswith("FFT")
    kw = {"window": window, "n_peaks": n_peaks, "threshold": threshold,
          "max_len": max_len}

    tree1 = guide_tree(sixtuple_distance(seqs))
    aln = progressive_align(seqs, sc, tree1, kind, s_op, use_fft, **kw)
    tree_used, rounds = tree1, 0
    if method in ("FFT-NS-2", "NW-NS-2", "FFT-NS-i"):
        # the second pass: rebuild the tree from the first alignment
        tree2 = guide_tree(sixtuple_distance(aln))
        aln = progressive_align(seqs, sc, tree2, kind, s_op, use_fft, **kw)
        tree_used = tree2
    score = wsp_score(aln, sc, s_op)
    if method == "FFT-NS-i":
        aln, score, rounds = iterative_refine(aln, sc, tree_used, s_op,
                                              max_iterate, kind, use_fft,
                                              **kw)
    return RichResult(payload={
        "estimate": aln,
        "alignment": aln,
        "score": float(score),
        "method": method,
        "seq_type": kind,
        "length": len(aln[0]),
        "n": len(seqs),
        "s_a": sc["s_a"],
        "s_op": float(s_op),
        "matrix_mode": matrix,
        "tree": tree_used,
        "refine_rounds": rounds,
        "note": ("Katoh et al. 2002: the FFT finds homologous segments "
                 "and the residue DP is restricted to the sub-matrices "
                 "between their centres; NW-NS-* skip the FFT and "
                 "matrix='all_positive' is the paper's NW-AP-2 control. "
                 "The default raw matrix is built from the same Grantham "
                 "volume and polarity as the FFT, not the paper's "
                 "200-PAM JTT -- pass raw_matrix and freqs for that."),
    })


mafftalignment = mafft_alignment


def cheatsheet():
    return ("mafft: MAFFT (Katoh et al. 2002). Residues become Grantham "
            "volume/polarity vectors, c(k) = c_v(k) + c_p(k) is got by "
            "FFT as V1*(m).V2(m), a 30-site window over the top 20 peaks "
            "at 0.7/site gives homologous segments (merged, then cut at "
            "150), a segment DP arranges them, and the residue DP runs "
            "only between their centres. Equation 7 rescales any matrix "
            "so random sequence scores S_a and identity scores 1 + S_a; "
            "the gap penalty S_op{1 - [g_start + g_end]/2} is zero where "
            "the group already has that gap. method= FFT-NS-1, FFT-NS-2, "
            "FFT-NS-i, NW-NS-1, NW-NS-2.")
