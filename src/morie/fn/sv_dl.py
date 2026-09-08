# morie.fn -- function file (rootcoder007/morie)
r"""DELLY: structural variant discovery from paired-end and split reads.

Rausch, T., Zichner, T., Schlattl, A., Stütz, A. M., Benes, V., &
Korbel, J. O. (2012) "DELLY: structural variant discovery by integrated
paired-end and split-read analysis", *Bioinformatics* 28(18), i333-i339.
doi:10.1093/bioinformatics/bts378

Two components, run in that order (the paper's Figure 1).

**Paired-end mapping analysis** (Section 2.1). Each library has a default
read-pair orientation and an insert size distribution; a pair is
*discordant* if its orientation is wrong or its insert size is more than
``n_sd`` standard deviations above the median (three, by default). Each
class of rearrangement leaves its own signature:

===================  ==========================================
deletion             default orientation, insert far too large
tandem duplication   the two reads swapped order but kept the
                     strands the default orientation induces
inversion            one read's orientation flipped; left- and
                     right-spanning pairs are clustered apart
translocation        the mates land on different chromosomes;
                     four types, by whether the chromosomes are
                     in sorted order and whether they are
                     inverted relative to each other
===================  ==========================================

Discordant pairs of one signature become the nodes of an undirected
weighted graph :math:`G(V,E)`. An edge joins two pairs that could support
the *same* breakpoint -- same orientation, and left and right ends within
the expected insert range -- and carries the weight

.. math::

   w(e_{v_i v_j}) = \bigl| \text{size}(p_i) - \text{size}(p_j) \bigr|,

the disagreement between the SV sizes the two pairs imply (for
translocations, where no size is defined, the sum of the absolute
differences of the two left-most alignment positions).

Under ideal conditions each rearrangement is one fully connected
component. It never is, so DELLY does not take components as calls:
within each component it grows a clique from the lowest-weight edge,

.. math::

   e_{\min} = \operatorname*{argmin}_{e \in E_{C_i}} w(e),

and repeatedly adds the lowest-weight edge with exactly one endpoint
already in the clique, provided the result is still a clique. When no
such edge exists the clique is maximal *for that seed* and is reported.
Singletons are dropped by construction. The interval comes from the
cluster: for a deletion, the start is the largest left-read end and the
end the smallest right-read start, so the call is the intersection all
supporting pairs agree on.

**Split-read analysis** (Section 2.2), which is what turns an interval
into a base. Every non-deletion type is first rewritten so that a plain
"deletion-type" search works (Figure 4): a tandem duplication has its
prefix and suffix swapped, an inversion has its second half reverse
complemented, a translocation gets both. Then, per candidate SV:

1. index the reference region by :math:`k`-mers (:math:`k = 7`), map each
   :math:`k`-mer of a read and bin the hits by alignment diagonal
   (reference position minus offset in the read);
2. take diagonals in decreasing hit count, marking each read
   :math:`k`-mer used once so it is counted for its best diagonal only;
   drop diagonals under ``k_min`` hits (three); drop the read unless two
   survive *and* those two hold at least half of its :math:`k`-mers --
   which is why a non-template insertion can only be found if it is
   shorter than half a read;
3. the gap between two consecutive diagonals is the SV size that read
   implies; take the best supported offset over all reads and keep the
   reads that agree with it;
4. build a gapless consensus by majority vote per column,
   :math:`c_i = \operatorname{argmax}_{\alpha} |\{r_{i,j} = \alpha\}|`;
5. align that consensus to the region with two Gotoh matrices, forward
   and reverse, giving :math:`f_i` (best score for the prefix
   :math:`c_1 \dots c_i`) and :math:`r_j` (best score for the suffix
   :math:`c_n \dots c_j`), and split at

   .. math::

      (\text{left}, \text{right}) =
      \operatorname*{argmax}_{i,j}\, f_i + r_j
      \quad\text{with } i < j .

   Not :math:`j = i + 1`: the slack is where a non-template
   microinsertion goes.
6. accept the refinement only if the split-read length agrees with the
   paired-end estimate to within ``max_length_diff`` (10%).

The alignment is global rather than local, with affine gaps, as the
paper specifies -- the consensus came from the reads, so it should align
end to end.

Everything here works on sequences and coordinates in memory rather than
BAM files; a pair is a dict with ``chrom1/pos1/strand1`` and the same for
mate 2, plus read lengths.
"""

import math  # noqa: F401  (kept for parity with the rest of morie.fn)

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = [
    "structural_variant",
    "sv_delly",
    "insert_size_stats",
    "classify_pair",
    "build_sv_graph",
    "maximal_clique",
    "paired_end_calls",
    "deletion_type_reference",
    "kmer_diagonals",
    "split_read_consensus",
    "gotoh_score_vectors",
    "optimal_split",
    "refine_breakpoint",
]

_SV_TYPES = ("DEL", "DUP", "INV", "TRA")
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


# ------------------------------------------------------------- helpers

def _pair(p):
    """Normalise one read pair; mate 1 is the left-most alignment."""
    try:
        c1, p1, s1 = p["chrom1"], int(p["pos1"]), p["strand1"]
        c2, p2, s2 = p["chrom2"], int(p["pos2"]), p["strand2"]
    except (KeyError, TypeError):
        raise ValueError("sv_dl: a pair needs chrom1/pos1/strand1 and "
                         "chrom2/pos2/strand2")
    l1 = int(p.get("len1", p.get("read_length", 100)))
    l2 = int(p.get("len2", p.get("read_length", 100)))
    if l1 < 1 or l2 < 1:
        raise ValueError("sv_dl: read lengths must be positive")
    if p1 < 0 or p2 < 0:
        raise ValueError("sv_dl: alignment positions must be non-negative")
    if s1 not in ("+", "-") or s2 not in ("+", "-"):
        raise ValueError("sv_dl: strands must be '+' or '-'")
    if (c2, p2) < (c1, p1):
        c1, p1, s1, l1, c2, p2, s2, l2 = c2, p2, s2, l2, c1, p1, s1, l1
    return {"chrom1": c1, "pos1": p1, "strand1": s1, "len1": l1,
            "chrom2": c2, "pos2": p2, "strand2": s2, "len2": l2,
            "seq": p.get("seq"), "id": p.get("id")}


def _median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        raise ValueError("sv_dl: no values to take a median of")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _insert(p):
    """Outer distance: left-most start to right-most end."""
    return (p["pos2"] + p["len2"]) - p["pos1"]


def insert_size_stats(pairs, orientation=None, spread="mad"):
    """Median and spread of the library insert size, and its orientation.

    The orientation is taken from the majority of same-chromosome pairs
    unless it is given, exactly as DELLY computes it per BAM file.

    Section 2.1 says the library is characterised by "the median and
    standard deviation", and the discordance cutoff is three of those
    standard deviations. Taken literally that is circular: the pairs that
    span a deletion are in the library too, and they are exactly the
    large-insert outliers, so they inflate the SD that is supposed to
    find them. On a simulated 300 bp deletion at 15x the plain SD comes
    out at 108 against a true 25, pushing the cutoff past most of the
    spanning pairs and costing the call four fifths of its support.

    ``spread="mad"`` (the default) therefore uses the median absolute
    deviation scaled by 1.4826, which is the same quantity for a Gaussian
    library and is not moved by the outliers. ``spread="sd"`` gives the
    literal reading.
    """
    ps = [_pair(p) for p in pairs]
    same = [p for p in ps if p["chrom1"] == p["chrom2"]]
    if not same:
        raise ValueError("sv_dl: no same-chromosome pairs to estimate the "
                         "insert size distribution from")
    if orientation is None:
        counts = {}
        for p in same:
            key = (p["strand1"], p["strand2"])
            counts[key] = counts.get(key, 0) + 1
        orientation = max(sorted(counts), key=lambda k: counts[k])
    orientation = tuple(orientation)
    if orientation[0] not in ("+", "-") or orientation[1] not in ("+", "-"):
        raise ValueError("sv_dl: orientation must be a pair of strands")
    concordant = [_insert(p) for p in same
                  if (p["strand1"], p["strand2"]) == orientation]
    if not concordant:
        raise ValueError("sv_dl: no pairs in the default orientation")
    if spread not in ("mad", "sd"):
        raise ValueError("sv_dl: spread must be 'mad' or 'sd'")
    med = _median(concordant)
    if len(concordant) > 1:
        if spread == "mad":
            sd = 1.4826 * _median([abs(v - med) for v in concordant])
        else:
            mu = sum(concordant) / float(len(concordant))
            var = sum((v - mu) ** 2 for v in concordant) / \
                (len(concordant) - 1.0)
            sd = math.sqrt(var)
    else:
        sd = 0.0
    return {"median": float(med), "sd": float(sd), "spread": spread,
            "orientation": orientation, "n": len(concordant)}


def classify_pair(p, median, sd, orientation=("+", "-"), n_sd=3.0):
    """The signature of one pair (Section 2.1, Figure 2).

    Returns ``None`` for a concordant pair, otherwise a
    ``(type, subtype)`` label -- ``("DEL", "")``, ``("DUP", "")``,
    ``("INV", "left"|"right")`` or ``("TRA", "0".."3")``.
    """
    p = _pair(p)
    d1, d2 = orientation
    if p["chrom1"] != p["chrom2"]:
        # four translocation classes: chromosomes already in sorted order
        # is guaranteed by _pair, so the class is fixed by which strands
        # departed from the library orientation
        t = 2 * (1 if p["strand1"] != d1 else 0) + \
            (1 if p["strand2"] != d2 else 0)
        return ("TRA", str(t))
    s1, s2 = p["strand1"], p["strand2"]
    if s1 == s2:
        # one read flipped: an inversion. Left- and right-spanning pairs
        # are clustered separately, so they are different subtypes.
        return ("INV", "left" if s1 == d1 else "right")
    if (s1, s2) == (d2, d1):
        # the reads kept their strands but swapped order: tandem duplication
        return ("DUP", "")
    if (s1, s2) == (d1, d2):
        if _insert(p) > median + n_sd * sd:
            return ("DEL", "")
        return None
    return None


def _sv_size(p, label, median):
    """The SV size this pair implies, used as the clustering weight."""
    if label[0] == "TRA":
        return None
    return _insert(p) - median


def build_sv_graph(pairs, median, sd, label, orientation=("+", "-"),
                   n_sd=3.0, window=None):
    """Nodes are pairs of one signature; edges join pairs that agree.

    An edge requires that both left and right ends are within the
    expected insert range; its weight is the disagreement between the
    implied SV sizes (for translocations, the summed shift of the two
    left-most positions).
    """
    if window is None:
        window = median + n_sd * sd
    ps = [_pair(p) for p in pairs]
    sizes = [_sv_size(p, label, median) for p in ps]
    edges = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            a, b = ps[i], ps[j]
            if a["chrom1"] != b["chrom1"] or a["chrom2"] != b["chrom2"]:
                continue
            if abs(a["pos1"] - b["pos1"]) > window:
                continue
            if abs(a["pos2"] - b["pos2"]) > window:
                continue
            if label[0] == "TRA":
                w = abs(a["pos1"] - b["pos1"]) + abs(a["pos2"] - b["pos2"])
            else:
                w = abs(sizes[i] - sizes[j])
            edges.append((float(w), i, j))
    return {"nodes": ps, "edges": edges, "sizes": sizes, "label": label}


def _components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for v in groups.values() if len(v) > 1]


def maximal_clique(members, edges):
    """Grow a clique from the lowest-weight edge (Section 2.1).

    ``edges`` is a list of ``(weight, i, j)``. The seed is
    :math:`e_{\\min}`; the clique then takes, repeatedly, the
    lowest-weight edge with exactly one endpoint inside it whose other
    endpoint is adjacent to every member. Returns the member indices.
    """
    keep = set(members)
    sub = sorted((w, i, j) for w, i, j in edges
                 if i in keep and j in keep)
    if not sub:
        return []
    adj = {}
    for w, i, j in sub:
        adj.setdefault(i, {})[j] = w
        adj.setdefault(j, {})[i] = w
    clique = set(sub[0][1:3])
    while True:
        best = None
        for w, i, j in sub:
            inside = (i in clique) + (j in clique)
            if inside != 1:
                continue
            outside = j if i in clique else i
            if all(outside in adj and m in adj[outside] for m in clique):
                if best is None or w < best[0]:
                    best = (w, outside)
        if best is None:
            break
        clique.add(best[1])
    return sorted(clique)


def paired_end_calls(pairs, median=None, sd=None, orientation=None,
                     n_sd=3.0, min_support=2, window=None, spread="mad"):
    """Cluster the discordant pairs into paired-end SV calls."""
    ps = [_pair(p) for p in pairs]
    if not ps:
        raise ValueError("sv_dl: no read pairs given")
    if median is None or sd is None or orientation is None:
        st = insert_size_stats(ps, orientation, spread)
        median = st["median"] if median is None else median
        sd = st["sd"] if sd is None else sd
        orientation = st["orientation"] if orientation is None \
            else tuple(orientation)
    if n_sd < 0:
        raise ValueError("sv_dl: n_sd must be non-negative")
    if min_support < 1:
        raise ValueError("sv_dl: min_support must be at least 1")
    by_label = {}
    for p in ps:
        lab = classify_pair(p, median, sd, orientation, n_sd)
        if lab is not None:
            by_label.setdefault(lab, []).append(p)
    calls = []
    for lab in sorted(by_label):
        group = by_label[lab]
        g = build_sv_graph(group, median, sd, lab, orientation, n_sd, window)
        for comp in _components(len(g["nodes"]), g["edges"]):
            members = maximal_clique(comp, g["edges"])
            if len(members) < min_support:
                continue
            sel = [g["nodes"][i] for i in members]
            start = max(p["pos1"] + p["len1"] for p in sel)
            end = min(p["pos2"] for p in sel)
            size = None
            if lab[0] != "TRA":
                size = sum(g["sizes"][i] for i in members) / len(members)
            calls.append({
                "type": lab[0],
                "subtype": lab[1],
                "chrom": sel[0]["chrom1"],
                "chrom2": sel[0]["chrom2"],
                "start": int(start),
                "end": int(end),
                "size": None if size is None else float(size),
                "support": len(members),
                "pairs": [dict(p) for p in sel],
                "precise": False,
            })
    calls.sort(key=lambda c: (c["type"], c["chrom"], c["start"]))
    return calls


# ------------------------------------------------------- split reads

def _revcomp(s):
    return "".join(_COMPLEMENT.get(c, "N") for c in reversed(s.upper()))


def deletion_type_reference(ref, sv_type):
    """Rewrite the region so a deletion-type search works (Figure 4).

    A tandem duplication has its two halves swapped, an inversion has
    its second half reverse complemented, a translocation gets both.
    """
    if sv_type not in _SV_TYPES:
        raise ValueError("sv_dl: sv_type must be one of %s" % (_SV_TYPES,))
    s = str(ref).upper()
    if sv_type == "DEL":
        return s
    h = len(s) // 2
    a, b = s[:h], s[h:]
    if sv_type == "DUP":
        return b + a
    if sv_type == "INV":
        return a + _revcomp(b)
    return _revcomp(b) + a


def kmer_diagonals(read, ref, k=7, k_min=3, require_half=True):
    """Bin the read's k-mer hits by alignment diagonal (Section 2.2).

    Diagonals are taken in decreasing hit count and each read k-mer is
    charged to its best diagonal only. Diagonals under ``k_min`` hits are
    dropped, and the read is rejected unless two survive holding at least
    half its k-mers. Returns the surviving ``(diagonal, hits)`` sorted by
    position in the read, or ``None``.
    """
    r, g = str(read).upper(), str(ref).upper()
    k = int(k)
    if k < 1:
        raise ValueError("sv_dl: k must be at least 1")
    if len(r) < k:
        return None
    index = {}
    for pos in range(len(g) - k + 1):
        km = g[pos:pos + k]
        if "N" in km:
            continue
        index.setdefault(km, []).append(pos)
    per_diag = {}
    total = 0
    for off in range(len(r) - k + 1):
        km = r[off:off + k]
        if "N" in km:
            continue
        total += 1
        for pos in index.get(km, ()):
            per_diag.setdefault(pos - off, []).append(off)
    if not per_diag:
        return None
    used, kept = set(), []
    for diag in sorted(per_diag, key=lambda d: (-len(per_diag[d]), d)):
        offs = [o for o in per_diag[diag] if o not in used]
        if len(offs) < int(k_min):
            continue
        used.update(offs)
        kept.append((diag, len(offs), min(offs)))
    if len(kept) < 2:
        return None
    top2 = sum(h for _, h, _ in sorted(kept, key=lambda t: -t[1])[:2])
    if require_half and total and top2 * 2 < total:
        return None
    kept.sort(key=lambda t: t[2])                  # order along the read
    return [(d, h) for d, h, _ in kept]


def split_read_consensus(reads, starts=None):
    """Gapless majority-vote consensus over the aligned reads.

    ``starts`` places each read in a common frame -- in DELLY that frame
    is the SV region and the offset is the read's first diagonal, since
    reads supporting one breakpoint hit it at different points along
    their length. Without it the reads are taken as already flush.
    Returns ``(consensus, start)``.
    """
    rs = [str(r).upper() for r in reads if r]
    if not rs:
        raise ValueError("sv_dl: no reads to build a consensus from")
    if starts is None:
        st = [0] * len(rs)
    else:
        st = [int(v) for v in starts]
        if len(st) != len(rs):
            raise ValueError("sv_dl: one start per read is required")
    lo = min(st)
    hi = max(a + len(r) for a, r in zip(st, rs))
    out = []
    for col in range(lo, hi):
        counts = {}
        for a, r in zip(st, rs):
            if a <= col < a + len(r):
                base = r[col - a]
                counts[base] = counts.get(base, 0) + 1
        if not counts:
            break                      # the consensus stays contiguous
        out.append(max(sorted(counts), key=lambda x: counts[x]))
    return "".join(out), lo


def _gotoh(query, ref, match=1.0, mismatch=-2.0, gap_open=-4.0,
           gap_extend=-1.0):
    """Affine-gap DP; returns, for each query prefix, its best score.

    The query must be aligned from its start but may end anywhere in the
    reference, which is what makes ``f_i`` "the best prefix alignment".
    """
    q, g = query, ref
    n, m = len(q), len(g)
    neg = float("-inf")
    # M: ends aligned; I: gap in the reference; D: gap in the query
    Mrow = [neg] * (m + 1)
    Irow = [neg] * (m + 1)
    Drow = [0.0] + [0.0] * m          # free leading gap in the query
    Mrow[0] = 0.0
    best, best_at = [], []
    for i in range(1, n + 1):
        nM = [neg] * (m + 1)
        nI = [neg] * (m + 1)
        nD = [neg] * (m + 1)
        nI[0] = max(Mrow[0] + gap_open, Irow[0] + gap_extend)
        for j in range(1, m + 1):
            s = match if q[i - 1] == g[j - 1] else mismatch
            prev = max(Mrow[j - 1], Irow[j - 1], Drow[j - 1])
            nM[j] = prev + s
            nI[j] = max(Mrow[j] + gap_open, Irow[j] + gap_extend)
            nD[j] = max(nM[j - 1] + gap_open, nD[j - 1] + gap_extend)
        Mrow, Irow, Drow = nM, nI, nD
        col = [max(Mrow[j], Irow[j], Drow[j]) for j in range(m + 1)]
        bj = max(range(m + 1), key=lambda j: col[j])
        best.append(col[bj])
        best_at.append(bj)
    return best, best_at


def gotoh_score_vectors(consensus, ref, match=1.0, mismatch=-2.0,
                        gap_open=-4.0, gap_extend=-1.0):
    r"""The paper's :math:`f` and :math:`r` (Section 2.2).

    :math:`f_i` is the best score for the prefix :math:`c_1 \dots c_i`
    against the reference; :math:`r_j` the best for the suffix
    :math:`c_n \dots c_j` against the reversed reference. Returns
    ``(f, f_end, r, r_start)`` where the two position vectors record
    where each best alignment ended in the reference, so a breakpoint can
    be translated back.
    """
    c, g = str(consensus).upper(), str(ref).upper()
    if not c or not g:
        raise ValueError("sv_dl: consensus and reference must be non-empty")
    f, f_at = _gotoh(c, g, match, mismatch, gap_open, gap_extend)
    rb, rb_at = _gotoh(c[::-1], g[::-1], match, mismatch, gap_open,
                       gap_extend)
    n, m = len(c), len(g)
    # rb[t] is the suffix of length t+1, i.e. it starts at c_{n-t}
    r = [0.0] * n
    r_at = [0] * n
    for t in range(n):
        r[n - 1 - t] = rb[t]
        r_at[n - 1 - t] = m - rb_at[t]
    return f, f_at, r, r_at


def optimal_split(f, r):
    r"""``argmax_{i<j} f_i + r_j`` -- the split with a microinsertion gap.

    Indices are 1-based over the consensus, as in the paper, so
    ``(i, j) = (4, 5)`` means the break falls between consensus bases 4
    and 5 with no inserted sequence.
    """
    n = len(f)
    if n != len(r):
        raise ValueError("sv_dl: f and r must have the same length")
    if n < 2:
        raise ValueError("sv_dl: the consensus is too short to split")
    best = None
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            v = f[i - 1] + r[j - 1]
            if best is None or v > best[0]:
                best = (v, i, j)
    return best[1], best[2], best[0]


def refine_breakpoint(call, reference, reads, k=7, k_min=3,
                      min_split_support=2, max_length_diff=0.10,
                      match=1.0, mismatch=-2.0, gap_open=-4.0,
                      gap_extend=-1.0):
    """Take one paired-end call to single-nucleotide resolution.

    ``reference`` is the SV region as a string, ``reads`` the candidate
    split reads. Returns ``None`` if the read support or the length check
    fails -- the call then stays imprecise rather than being invented.
    """
    region = deletion_type_reference(reference, call["type"])
    offsets = {}
    per_read = {}
    first_diag = {}
    for idx, rd in enumerate(reads):
        diags = kmer_diagonals(rd, region, k, k_min)
        if diags is None:
            continue
        first_diag[idx] = diags[0][0]
        # the gap between consecutive diagonals is the size this read implies
        for a in range(len(diags) - 1):
            off = diags[a + 1][0] - diags[a][0]
            if off == 0:
                continue
            offsets[off] = offsets.get(off, 0) + 1
            per_read.setdefault(off, []).append(idx)
    if not offsets:
        return None
    best_off = max(sorted(offsets), key=lambda o: offsets[o])
    support = per_read[best_off]
    if len(support) < int(min_split_support):
        return None
    starts = [first_diag[i] for i in support]
    consensus, _ = split_read_consensus([reads[i] for i in support], starts)
    f, f_at, r, r_at = gotoh_score_vectors(consensus, region, match,
                                           mismatch, gap_open, gap_extend)
    i, j, score = optimal_split(f, r)
    left_ref = f_at[i - 1]
    right_ref = r_at[j - 1]
    size = right_ref - left_ref
    if call["size"] is not None and call["size"] > 0:
        if abs(size - call["size"]) > max_length_diff * abs(call["size"]):
            return None
    # Microhomology: bases shared by the two breakpoint flanks. The
    # junction slides freely across them, so a call inside the homology
    # is not wrong -- it describes the same haplotype. DELLY's alignment
    # places it at the left end, and the length is reported so a caller
    # can see how much room there was.
    # The alignment places the junction as far left as it will go, so the
    # room left over runs to the right: while the base after the left
    # breakpoint matches the base after the right one, the cut can slide
    # by one and describe the same haplotype.
    hom = 0
    while (left_ref + hom < len(region) and right_ref + hom < len(region) and
           region[left_ref + hom] == region[right_ref + hom]):
        hom += 1
    return {"start": int(left_ref), "end": int(right_ref),
            "size": int(size), "split_support": len(support),
            "consensus": consensus, "score": float(score),
            "microinsertion": consensus[i:j - 1],
            "microhomology": int(hom),
            "kmer_offset": int(best_off)}


# ------------------------------------------------------------- driver

def structural_variant(pairs, reference=None, split_reads=None,
                       orientation=None, median=None, sd=None, n_sd=3.0,
                       min_support=2, k=7, k_min=3, min_split_support=2,
                       max_length_diff=0.10, window=None, spread="mad"):
    """Call structural variants from read pairs, refined by split reads.

    ``pairs`` is a sequence of dicts with ``chrom1/pos1/strand1`` and
    ``chrom2/pos2/strand2`` (plus ``len1``/``len2``). Give ``reference``
    (the sequence of the contig) and ``split_reads`` to get
    single-nucleotide breakpoints; without them the calls come back at
    paired-end resolution with ``precise=False``.
    """
    calls = paired_end_calls(pairs, median, sd, orientation, n_sd,
                             min_support, window, spread)
    st = insert_size_stats(pairs, orientation, spread)
    refined = 0
    if reference is not None and split_reads:
        for call in calls:
            lo = max(0, call["start"] - int(st["median"]))
            hi = min(len(reference), call["end"] + int(st["median"]))
            region = str(reference)[lo:hi]
            if not region:
                continue
            got = refine_breakpoint(call, region, split_reads, k, k_min,
                                    min_split_support, max_length_diff)
            if got is None:
                continue
            call["start"] = got["start"] + lo
            call["end"] = got["end"] + lo
            call["size"] = float(got["size"])
            call["split_support"] = got["split_support"]
            call["consensus"] = got["consensus"]
            call["microinsertion"] = got["microinsertion"]
            call["microhomology"] = got["microhomology"]
            call["precise"] = True
            refined += 1
    return RichResult(payload={
        "estimate": calls,
        "calls": calls,
        "n_calls": len(calls),
        "n_precise": refined,
        "insert_median": st["median"],
        "insert_sd": st["sd"],
        "spread": spread,
        "orientation": st["orientation"],
        "n_sd": float(n_sd),
        "min_support": int(min_support),
        "method": ("DELLY (Rausch et al. 2012): discordant paired-end "
                   "clustering by maximal clique, refined by k-mer "
                   "split-read search and a double-dynamic-programming "
                   "split alignment"),
        "note": ("calls are imprecise (paired-end resolution) unless a "
                 "reference and split reads are supplied and the "
                 "split-read length agrees with the paired-end estimate "
                 "to within max_length_diff"),
    })


sv_delly = structural_variant


def cheatsheet():
    return ("sv_dl: DELLY (Rausch et al. 2012). Discordant pairs are "
            "typed by orientation and insert size (DEL, DUP, INV "
            "left/right, four TRA classes), made into a weighted graph "
            "where the weight is the disagreement in implied SV size, "
            "and each component yields a maximal clique grown from its "
            "lowest-weight edge. Split reads are then found by k-mer "
            "diagonal counting, a majority-vote consensus is built, and "
            "forward and reverse Gotoh score vectors are split at "
            "argmax_{i<j} f_i + r_j to give the breakpoint to a base.")
