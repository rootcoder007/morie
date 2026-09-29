# morie.fn -- function file (rootcoder007/morie)
"""Sequence tools: greedy overlap-layout-consensus assembly of reads with errors, and FoldIndex prediction of
intrinsically disordered protein regions from charge and hydropathy."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["olc_assembly", "protein_disorder"]


def _best_overlap(a, b, min_overlap, max_error):
    for L in range(min(len(a), len(b)) - 1, min_overlap - 1, -1):
        mm = 0
        lim = int(max_error * L)
        ok = True
        for t in range(L):
            if a[len(a) - L + t] != b[t]:
                mm += 1
                if mm > lim:
                    ok = False
                    break
        if ok:
            return L, mm
    return 0, 0


def olc_assembly(long_reads, *, min_overlap: int = 3, max_error: float = 0.0) -> RichResult:
    r"""Overlap-layout-consensus assembly with greedy layout and column-majority consensus.

    *Overlap*: after removing duplicate and contained reads, every ordered
    pair ``(i, j)`` gets its longest suffix-prefix overlap ``L >=
    min_overlap`` with at most ``floor(max_error L)`` mismatches.
    *Layout*: overlaps are accepted greedily, longest first (then fewest
    mismatches, then read order), when ``i`` has no successor yet, ``j`` no
    predecessor, and the edge closes no cycle; each resulting path places
    read ``j`` at offset ``offset_i + len_i - L``. *Consensus*: each column
    takes the most frequent base among the reads covering it (ties to the
    alphabetically first). Returns ``contigs`` (longest first), each
    contig's ``layouts`` (read index, offset) and the ``overlaps`` used.

    References
    ----------
    Staden, R. (1980). A new computer method for the storage and
    manipulation of DNA gel reading data. *Nucleic Acids Research*, 8,
    3673-3694.
    Myers, E. W. (2005). The fragment assembly string graph.
    *Bioinformatics*, 21 (suppl. 2), ii79-ii85.
    Compeau, P. and Pevzner, P. (2014). *Bioinformatics Algorithms*, vol. 1,
    ch. 3. Active Learning Publishers.

    Examples
    --------
    >>> olc_assembly(["ATGGCGT", "GCGTGCA", "TGCAATG", "CAATGGA"]).contigs
    ['ATGGCGTGCAATGGA']
    """
    reads = [str(r).upper() for r in long_reads]
    keep = []
    for i, r in enumerate(reads):
        dup = any(reads[j] == r for j in range(i))
        cont = any(len(s) > len(r) and r in s for s in reads)
        if not dup and not cont:
            keep.append(i)
    edges = []
    for i in keep:
        for j in keep:
            if i != j:
                L, mm = _best_overlap(reads[i], reads[j], int(min_overlap), float(max_error))
                if L:
                    edges.append((L, mm, i, j))
    edges.sort(key=lambda e: (-e[0], e[1], e[2], e[3]))
    succ, pred = {}, {}
    comp = {i: i for i in keep}

    def find(x):
        while comp[x] != x:
            x = comp[x]
        return x

    used = []
    for L, mm, i, j in edges:
        if i in succ or j in pred or find(i) == find(j):
            continue
        succ[i] = (j, L)
        pred[j] = i
        comp[find(j)] = find(i)
        used.append((i, j, L, mm))
    contigs = []
    for s in keep:
        if s in pred:
            continue
        lay, off, cur = [], 0, s
        while True:
            lay.append((cur, off))
            if cur not in succ:
                break
            nxt, L = succ[cur]
            off += len(reads[cur]) - L
            cur = nxt
        width = max(o + len(reads[k]) for k, o in lay)
        seq = []
        for col in range(width):
            counts = {}
            for k, o in lay:
                if o <= col < o + len(reads[k]):
                    ch = reads[k][col - o]
                    counts[ch] = counts.get(ch, 0) + 1
            seq.append(min(counts, key=lambda c: (-counts[c], c)))
        contigs.append(("".join(seq), lay))
    contigs.sort(key=lambda c: (-len(c[0]), c[0]))
    return RichResult(
        payload={
            "contigs": [c[0] for c in contigs],
            "layouts": [c[1] for c in contigs],
            "overlaps": used,
            "n_reads_used": len(keep),
        }
    )


_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}  # fmt: skip
_CHARGE = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}


def _fold_index(seq):
    h = 0.0
    q = 0.0
    for a in seq:
        h += (_KD[a] + 4.5) / 9.0
        q += _CHARGE.get(a, 0.0)
    n = len(seq)
    return 2.785 * (h / n) - abs(q / n) - 1.151, h / n, q / n


def protein_disorder(sequence, *, window: int = 51, min_region: int = 5) -> RichResult:
    r"""FoldIndex disorder prediction (Prilusky et al. 2005) from the Uversky charge-hydropathy boundary.

    ``FoldIndex = 2.785 <H> - |<R>| - 1.151`` with ``<H>`` the mean
    Kyte-Doolittle hydropathy rescaled to ``[0, 1]`` (``(KD + 4.5) / 9``)
    and ``<R>`` the mean net charge (K, R = +1; D, E = -1). The global
    index uses the whole chain; the residue profile evaluates a sliding
    ``window`` (odd) and assigns each window's value to its centre, the
    ``window // 2`` residues at each end taking the nearest full window.
    Residues with ``FoldIndex < 0`` are predicted disordered; runs of at
    least ``min_region`` are reported as 1-based inclusive regions.

    References
    ----------
    Uversky, V. N., Gillespie, J. R. and Fink, A. L. (2000). Why are
    "natively unfolded" proteins unstructured under physiologic
    conditions? *Proteins*, 41, 415-427.
    Prilusky, J. et al. (2005). FoldIndex: a simple tool to predict whether
    a given protein sequence is intrinsically unfolded. *Bioinformatics*,
    21, 3435-3438.
    Kyte, J. and Doolittle, R. F. (1982). A simple method for displaying
    the hydropathic character of a protein. *Journal of Molecular
    Biology*, 157, 105-132.

    Examples
    --------
    >>> r = protein_disorder("MKKEEEKKPSEESKEDKKSEEGAPLLIVAVLLFAGIVLLVAWFILK", window=11)
    >>> round(r.fold_index, 6), r.regions
    (0.230524, [[1, 21]])
    """
    seq = "".join(ch for ch in str(sequence).upper() if not ch.isspace())
    bad = sorted({a for a in seq if a not in _KD})
    if bad:
        raise ValueError(f"unknown residues: {''.join(bad)}")
    n = len(seq)
    if n == 0:
        raise ValueError("empty sequence")
    w = int(window)
    if w % 2 == 0 or w < 1:
        raise ValueError("window must be a positive odd integer")
    fi, H, R = _fold_index(seq)
    if n <= w:
        prof = [fi] * n
    else:
        half = w // 2
        cen = [_fold_index(seq[s : s + w])[0] for s in range(n - w + 1)]
        prof = [cen[min(max(i - half, 0), n - w)] for i in range(n)]
    dis = [v < 0 for v in prof]
    regions, i = [], 0
    while i < n:
        if dis[i]:
            j = i
            while j + 1 < n and dis[j + 1]:
                j += 1
            if j - i + 1 >= int(min_region):
                regions.append([i + 1, j + 1])
            i = j + 1
        else:
            i += 1
    return RichResult(
        payload={
            "fold_index": fi,
            "mean_hydropathy": H,
            "mean_net_charge": R,
            "profile": prof,
            "disordered": dis,
            "regions": regions,
            "fraction_disordered": sum(1 for v in dis if v) / n,
        }
    )


def cheatsheet() -> str:
    return (
        "olc_assembly(reads) -> greedy OLC contigs; protein_disorder(seq) -> FoldIndex profile and disordered regions."
    )
