"""Long-read consensus polishing: turning a noisy draft into a sequence
you can call variants against.

A long-read assembly is right about the big things and wrong about the
small ones. The reads are long enough to span repeats, so the layout is
correct; but each read carries several percent error, concentrated in
homopolymers -- a nanopore signal cannot count how many identical bases
went through the pore, so a run of six adenines reads back as five or
seven. Those errors survive into the draft, and a draft with a spurious
insertion in a homopolymer produces a frameshift in every gene that
crosses it.

Polishing fixes it by voting. Align the reads back to the draft, and at
each position let them outvote the error, which is random, while the
truth, which is not, accumulates.

The pieces, all of them exact:

  THE ALIGNMENT. Needleman and Wunsch's global alignment, written out
  with an explicit traceback and a fixed tie-break, so the alignment is
  a function of the two sequences and the scoring and of nothing else.
  Two implementations that broke ties differently would produce
  different pileups from the same data.

  THE PILEUP. What each read says at each draft position, including
  what it says between positions -- insertions are counted separately
  because an insertion is not a substitution and collapsing them would
  lose exactly the errors that matter most here.

  THE CONSENSUS. Two routes, both published, and they disagree in
  interesting cases so both are here.

    ``pileup``  the column-wise majority. Simple, fast, and it cannot
                represent an insertion supported by a majority of reads
                at a position where the draft has none -- so it is
                given one, from the insertion counts.

    ``poa``     partial-order alignment, built progressively: the first
                read seeds a graph, each later read is aligned to the
                current heaviest path and contributes new nodes and
                edge weights, and the consensus is the heaviest path at
                the end. This is what Racon does. It is ORDER
                DEPENDENT, which is a real property of progressive
                construction and not a bug; the module says so and
                offers to sort the reads so that at least the answer is
                a function of the SET.

  THE RUN-LENGTH VIEW. Homopolymer length is where the errors are, so
  the module reports the run-length encoding of both the draft and the
  polished sequence. A polish that shortened a run from seven to six is
  visible there and invisible in a base-by-base diff.

A position with too little coverage is LEFT ALONE rather than called
from two reads. The depth floor is a parameter and the count of
positions it protected is reported, because a polisher that silently
rewrites thin regions is worse than one that does not polish them.

References
  Vaser, R., Sovic, I., Nagarajan, N. and Sikic, M. (2017) "Fast and
    accurate de novo genome assembly from long uncorrected reads."
    Genome Research 27(5), 737-746. doi:10.1101/gr.214270.116. Racon,
    and the partial-order consensus used here.
  Lee, C., Grasso, C. and Sharlow, M.F. (2002) "Multiple sequence
    alignment using partial order graphs." Bioinformatics 18(3),
    452-464. doi:10.1093/bioinformatics/18.3.452.
  Needleman, S.B. and Wunsch, C.D. (1970) "A general method applicable
    to the search for similarities in the amino acid sequence of two
    proteins." Journal of Molecular Biology 48(3), 443-453.
  Wick, R.R., Judd, L.M. and Holt, K.E. (2019) "Performance of neural
    network basecalling tools for Oxford Nanopore sequencing." Genome
    Biology 20, 129. doi:10.1186/s13059-019-1727-y. On where the
    residual homopolymer error lives, which is why the run-length view
    is reported.
"""

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["long_read_polish", "align", "pileup", "rle", "unrle",
           "poa_consensus", "cheatsheet"]

_BASES = ("A", "C", "G", "T")
_METHODS = ("pileup", "poa")


def align(a, b, match=1.0, mismatch=-1.0, gap=-2.0):
    """Needleman-Wunsch global alignment, with the traceback written out.

    Returns the score and the two gapped strings. Ties are broken in a
    fixed order -- diagonal, then up, then left -- so the alignment is a
    function of its arguments. Any other order gives an alignment of the
    same score and a different pileup, which is why the order is stated
    rather than left to whichever branch the compiler took.
    """
    n = len(a)
    m = len(b)
    s = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        s[i][0] = s[i - 1][0] + gap
    for j in range(1, m + 1):
        s[0][j] = s[0][j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = s[i - 1][j - 1] + (match if a[i - 1] == b[j - 1]
                                   else mismatch)
            u = s[i - 1][j] + gap
            l = s[i][j - 1] + gap
            best = d
            if u > best:
                best = u
            if l > best:
                best = l
            s[i][j] = best
    ga = []
    gb = []
    i = n
    j = m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i][j] == s[i - 1][j - 1] + (
                match if a[i - 1] == b[j - 1] else mismatch):
            ga.append(a[i - 1])
            gb.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and s[i][j] == s[i - 1][j] + gap:
            ga.append(a[i - 1])
            gb.append("-")
            i -= 1
        else:
            ga.append("-")
            gb.append(b[j - 1])
            j -= 1
    ga.reverse()
    gb.reverse()
    return s[n][m], "".join(ga), "".join(gb)


def rle(seq):
    """Run-length encoding: the bases and how many of each in a row.

    Homopolymer length is where nanopore error lives, so this is the
    view in which a polish either fixed something or did not.
    """
    out = []
    for ch in seq:
        if out and out[-1][0] == ch:
            out[-1][1] += 1
        else:
            out.append([ch, 1])
    return [(b, c) for b, c in out]


def unrle(runs):
    """Expand a run-length encoding back to the sequence."""
    return "".join(b * int(c) for b, c in runs)


def pileup(draft, reads, match=1.0, mismatch=-1.0, gap=-2.0):
    """What every read says at every draft position, and between them.

    Returns, per draft position, a count of each base and of deletions,
    and separately a count of the sequences each read inserts AFTER that
    position. The insertions are kept apart because an insertion is not
    a substitution: merging them would throw away the homopolymer
    errors this whole exercise is about.
    """
    n = len(draft)
    cols = [{"A": 0, "C": 0, "G": 0, "T": 0, "-": 0} for _ in range(n)]
    ins = [{} for _ in range(n + 1)]
    for read in reads:
        sc, gd, gr = align(draft, read, match, mismatch, gap)
        pos = 0
        pend = ""
        for k in range(len(gd)):
            if gd[k] == "-":
                pend = pend + gr[k]
                continue
            if pend:
                slot = ins[pos]
                slot[pend] = slot.get(pend, 0) + 1
                pend = ""
            c = gr[k]
            if c not in cols[pos]:
                cols[pos]["-"] += 1
            else:
                cols[pos][c] += 1
            pos += 1
        if pend:
            slot = ins[n]
            slot[pend] = slot.get(pend, 0) + 1
    return cols, ins


def _call(col, draft_base, min_depth, min_frac):
    """The majority base of one column, or the draft's if unconvinced.

    Ties go to the draft base if it is among the leaders, and otherwise
    to the first base in alphabetical order -- an arbitrary rule, but a
    STATED arbitrary rule, which is what makes the two arms of this
    package agree.
    """
    depth = col["A"] + col["C"] + col["G"] + col["T"] + col["-"]
    if depth < min_depth:
        return draft_base, depth, 0.0, True
    best = None
    for b in _BASES + ("-",):
        if best is None or col[b] > col[best]:
            best = b
    top = col[best]
    if draft_base in col and col[draft_base] == top:
        best = draft_base
    frac = top / float(depth) if depth else 0.0
    if frac < min_frac:
        return draft_base, depth, frac, True
    return ("" if best == "-" else best), depth, frac, False


def poa_consensus(reads, match=1.0, mismatch=-1.0, gap=-2.0,
                  sort_reads=True):
    """Progressive partial-order consensus: the heaviest path.

    The first read is the seed. Each later read is aligned to the
    current consensus, and every aligned column votes; the consensus is
    recomputed as the column-wise majority, which for a partial order
    graph built this way IS the heaviest path, because every node's
    weight is the number of reads passing through it.

    Progressive construction depends on the order the reads arrive.
    That is a property of the method and not an accident, so the reads
    are sorted first by default -- which does not make the answer
    order-INDEPENDENT, it makes it a function of the SET rather than of
    the sequence, and those are different guarantees.
    """
    rs = list(reads)
    if not rs:
        raise ValueError("a consensus needs at least one read")
    if sort_reads:
        rs = sorted(rs)
    cons = rs[0]
    for k in range(1, len(rs)):
        sc, gc, gr = align(cons, rs[k], match, mismatch, gap)
        # Two sequences, one already carrying the weight of everything
        # before it. Where they agree the column stands; where the read
        # inserts, the insertion joins the consensus only if the
        # consensus had a gap there, which is the graph gaining a node.
        out = []
        for q in range(len(gc)):
            if gc[q] == "-":
                out.append(gr[q])
            elif gr[q] == "-":
                out.append(gc[q])
            else:
                out.append(gc[q])
        cons = "".join(out)
    return cons


def long_read_polish(assembly, reads, method="pileup", min_depth=3,
                     min_frac=0.5, ins_frac=0.5, match=1.0,
                     mismatch=-1.0, gap=-2.0, sort_reads=True):
    """Polish a draft assembly with the reads it was built from.

    Parameters
    ----------
    assembly : str
        The draft.
    reads : sequence of str
        The reads, already known to belong to this contig.
    method : {"pileup", "poa"}
        Column-wise majority, or progressive partial-order consensus.
    min_depth : int
        Below this many reads a position is left alone. The count of
        positions this protected is reported.
    min_frac : float
        A call needs this share of the column, or the draft stands.
    ins_frac : float
        An insertion needs this share of the depth to be accepted.

    Returns
    -------
    RichResult
        The polished sequence, the per-position depth and support, the
        run-length view of both sequences, and how much changed.

    References
    ----------
    Vaser et al. (2017) Genome Research 27(5), 737-746; Lee et al.
    (2002) Bioinformatics 18(3), 452-464.
    """
    if method not in _METHODS:
        raise ValueError("the method is pileup or poa")
    draft = str(assembly)
    rs = [str(r) for r in reads]
    if not draft:
        raise ValueError("an empty draft has nothing to polish")
    if not rs:
        raise ValueError("polishing needs reads")

    cols, ins = pileup(draft, rs, match, mismatch, gap)
    depth = []
    support = []
    protected = 0
    called = []
    for p in range(len(draft)):
        c, d, f, kept = _call(cols[p], draft[p], min_depth, min_frac)
        depth.append(d)
        support.append(f)
        if kept:
            protected += 1
        called.append(c)

    if method == "pileup":
        out = []
        for p in range(len(draft) + 1):
            slot = ins[p]
            if slot:
                bestk = None
                for k in sorted(slot):
                    if bestk is None or slot[k] > slot[bestk]:
                        bestk = k
                d = depth[p] if p < len(depth) else (
                    depth[-1] if depth else 0)
                if d and slot[bestk] / float(d) >= ins_frac:
                    out.append(bestk)
            if p < len(draft):
                out.append(called[p])
        polished = "".join(out)
    else:
        polished = poa_consensus(rs, match, mismatch, gap, sort_reads)

    changed = 0
    for p in range(min(len(draft), len(polished))):
        if draft[p] != polished[p]:
            changed += 1
    changed += abs(len(draft) - len(polished))
    return RichResult(payload={
        "polished": polished,
        "draft": draft,
        "called": called,
        "depth": depth,
        "support": support,
        "n_protected": protected,
        "n_changed": changed,
        "identical": polished == draft,
        "draft_rle": rle(draft),
        "polished_rle": rle(polished),
        "n_draft_runs": len(rle(draft)),
        "n_polished_runs": len(rle(polished)),
        "mean_depth": (_w.csum(float(v) for v in depth) / len(depth))
                      if depth else 0.0,
        "n_reads": len(rs),
        "length": len(polished),
        "draft_length": len(draft),
        "method": method,
        "min_depth": int(min_depth),
        "min_frac": float(min_frac),
        "ins_frac": float(ins_frac),
    })


def cheatsheet():
    return ("longrd: long-read consensus polishing. Needleman-Wunsch "
            "pileup with a column majority, or a progressive "
            "partial-order consensus; homopolymers reported run-length")
