"""Metagenome assembly: rebuilding many genomes at once from reads that
do not say which organism they came from.

Assembling one genome is a graph problem. Assembling a community is the
same graph problem with the assumption that breaks it removed. A
single-genome assembler leans on uniform coverage: a stretch of the
graph seen far less often than the rest is an error, and can be cut
away. In a metagenome that reasoning destroys the data, because a
species at one percent abundance is genuinely covered a hundred times
less than the dominant one, and it is not an error -- it is the finding.

metaSPAdes' answer, and this module's, is to make every coverage test
RELATIVE. A dead end is removed when it is much thinner than the path
it hangs off, not when it is thin in absolute terms. A rare organism's
contigs are thin everywhere, so nothing about them is locally
anomalous, and they survive.

The pieces:

  THE DE BRUIJN GRAPH. Nodes are the k-1-mers, edges are the k-mers,
  and an edge's weight is how many times that k-mer was read. Exact,
  and the only place the reads are touched.

  THE UNITIGS. Maximal non-branching paths -- walk forward while the
  next node has exactly one way in and one way out. Where the walk has
  to stop is where the data genuinely stops determining the sequence,
  so a unitig is the longest thing that can be asserted rather than
  guessed. Circular components with no branch at all are found
  separately, because a cycle has no starting node to walk from and an
  assembler that only walked from branch points would silently drop
  every plasmid.

  THE CLEANING, both relative. A TIP is a short dead-end whose coverage
  is a small fraction of the branch it leaves; a BUBBLE is two paths
  between the same two nodes, which is what a single-base difference
  between two strains looks like, and the thinner side goes. Both
  thresholds are parameters and both counts are reported, because an
  assembler that cleaned silently would be indistinguishable from one
  that lost data.

Contigs come back sorted longest first, ties broken by sequence in byte
order, so the output is a function of the reads and the parameters and
of nothing else.

References
  Nurk, S., Meleshko, D., Korobeynikov, A. and Pevzner, P.A. (2017)
    "metaSPAdes: a new versatile metagenomic assembler." Genome
    Research 27(5), 824-834. doi:10.1101/gr.213959.116. The relative
    coverage treatment that makes this a metagenome assembler rather
    than a genome assembler run on a mixture.
  Pevzner, P.A., Tang, H. and Waterman, M.S. (2001) "An Eulerian path
    approach to DNA fragment assembly." Proceedings of the National
    Academy of Sciences 98(17), 9748-9753. The de Bruijn formulation.
  Bankevich, A., Nurk, S., Antipov, D., Gurevich, A.A., Dvorkin, M.,
    Kulikov, A.S., Lesin, V.M., Nikolenko, S.I., Pham, S., Prjibelski,
    A.D., Pyshkin, A.V., Sirotkin, A.V., Vyahhi, N., Tesler, G.,
    Alekseyev, M.A. and Pevzner, P.A. (2012) "SPAdes: a new genome
    assembly algorithm and its applications to single-cell
    sequencing." Journal of Computational Biology 19(5), 455-477.
    doi:10.1089/cmb.2012.0021. The tip and bubble machinery.
"""

from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["metagenome_assembly", "de_bruijn", "unitigs", "n50",
           "kmers", "cheatsheet"]


def kmers(seq, k):
    """Every k-mer of a sequence, in order, with repeats kept."""
    k = int(k)
    if k < 2:
        raise ValueError("a de Bruijn graph needs k of at least two")
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def de_bruijn(reads, k):
    """Nodes are the k-1-mers, edges the k-mers, weights the counts.

    Reads shorter than k contribute nothing and are counted rather than
    dropped in silence: a run where most reads are shorter than the
    chosen k has a k problem, and the caller should be told.
    """
    k = int(k)
    edges = {}
    short = 0
    used = 0
    for r in reads:
        s = str(r)
        if len(s) < k:
            short += 1
            continue
        used += 1
        for km in kmers(s, k):
            edges[km] = edges.get(km, 0) + 1
    out = {}
    inc = {}
    nodes = set()
    for km in sorted(edges):
        a = km[:-1]
        b = km[1:]
        nodes.add(a)
        nodes.add(b)
        out.setdefault(a, []).append(km)
        inc.setdefault(b, []).append(km)
    return {"edges": edges, "out": out, "in": inc,
            "nodes": sorted(nodes), "k": k, "short": short,
            "used": used}


def _outdeg(g, v):
    return len(g["out"].get(v, []))


def _indeg(g, v):
    return len(g["in"].get(v, []))


def _walk(g, first):
    """Follow a non-branching path forward from one edge."""
    path = [first]
    v = first[1:]
    while _outdeg(g, v) == 1 and _indeg(g, v) == 1:
        nxt = g["out"][v][0]
        if nxt in path:
            break
        path.append(nxt)
        v = nxt[1:]
    return path


def _seq(path):
    return path[0] + "".join(e[-1] for e in path[1:])


def unitigs(g):
    """Maximal non-branching paths, plus the pure cycles.

    A branch point is any node that is not exactly one-in one-out.
    Every walk starts at an edge leaving one. What is left over after
    that is made only of one-in one-out nodes, which means it is a
    cycle; those are picked up separately so a circular replicon does
    not vanish for want of a place to start.
    """
    seen = {}
    out = []
    for v in g["nodes"]:
        if _outdeg(g, v) > 0 and not (_outdeg(g, v) == 1
                                      and _indeg(g, v) == 1):
            for e in g["out"][v]:
                if e in seen:
                    continue
                p = _walk(g, e)
                for x in p:
                    seen[x] = True
                out.append(p)
    for km in sorted(g["edges"]):
        if km in seen:
            continue
        p = _walk(g, km)
        for x in p:
            seen[x] = True
        out.append(p)
    rows = []
    for p in out:
        w = [float(g["edges"][e]) for e in p]
        rows.append({
            "path": p,
            "seq": _seq(p),
            "start": p[0][:-1],
            "end": p[-1][1:],
            "length": len(_seq(p)),
            "coverage": _w.csum(w) / len(w),
            "n_edges": len(p),
        })
    rows.sort(key=lambda r: (-r["length"], r["seq"]))
    return rows


def n50(lengths):
    """The length at which half the assembly is in contigs that long.

    Sorted longest first, accumulate until the running total reaches
    half the assembly; the contig you were on is the answer. Reported
    for an empty assembly as zero rather than as an error, because an
    assembly with no contigs is a result and not a mistake.
    """
    ls = sorted((int(v) for v in lengths), reverse=True)
    total = 0
    for v in ls:
        total += v
    if total == 0:
        return 0
    run = 0
    for v in ls:
        run += v
        if run * 2 >= total:
            return v
    return ls[-1]


def _drop(g, es):
    for e in es:
        if e not in g["edges"]:
            continue
        del g["edges"][e]
    out = {}
    inc = {}
    nodes = set()
    for km in sorted(g["edges"]):
        a = km[:-1]
        b = km[1:]
        nodes.add(a)
        nodes.add(b)
        out.setdefault(a, []).append(km)
        inc.setdefault(b, []).append(km)
    g["out"] = out
    g["in"] = inc
    g["nodes"] = sorted(nodes)
    return g


def metagenome_assembly(reads, k, tip_length=None, tip_ratio=0.2,
                        bubble_ratio=0.5, rounds=2, min_length=None):
    """Assemble a community from its reads.

    Parameters
    ----------
    reads : sequence of str
        The reads.
    k : int
        The k-mer length.
    tip_length : int or None
        A dead-end shorter than this is a tip candidate. None is twice
        k, the usual choice, stated rather than hidden.
    tip_ratio : float
        A tip goes only if its coverage is below this fraction of the
        best branch it leaves -- RELATIVE, which is the whole point:
        an absolute cut-off deletes the rare organisms.
    bubble_ratio : float
        The thinner side of a bubble goes if it is below this fraction
        of the thicker.
    rounds : int
        Cleaning passes.
    min_length : int or None
        Contigs shorter than this are reported separately rather than
        thrown away.

    Returns
    -------
    RichResult
        The contigs longest first, their coverage, the N50, and what
        the cleaning removed.

    References
    ----------
    Nurk et al. (2017) Genome Research 27(5), 824-834; Bankevich et al.
    (2012) J. Comput. Biol. 19(5), 455-477.
    """
    k = int(k)
    if k < 2:
        raise ValueError("a de Bruijn graph needs k of at least two")
    rs = [str(r) for r in reads]
    if not rs:
        raise ValueError("an assembly needs reads")
    if tip_length is None:
        tip_length = 2 * k
    g = de_bruijn(rs, k)
    n_edges0 = len(g["edges"])
    tips = 0
    bubbles = 0
    for _ in range(int(rounds)):
        us = unitigs(g)
        by_start = {}
        for u in us:
            by_start.setdefault(u["start"] + "|" + u["end"],
                                []).append(u)
        drop = []
        for key in sorted(by_start):
            grp = by_start[key]
            if len(grp) < 2:
                continue
            grp = sorted(grp, key=lambda r: (-r["coverage"], r["seq"]))
            for r in grp[1:]:
                if r["coverage"] <= bubble_ratio * grp[0]["coverage"]:
                    drop.extend(r["path"])
                    bubbles += 1
        if drop:
            g = _drop(g, drop)
            continue
        drop = []
        for ui in range(len(us)):
            u = us[ui]
            dead = (_outdeg(g, u["end"]) == 0
                    or _indeg(g, u["start"]) == 0)
            if not dead or u["length"] >= tip_length:
                continue
            neigh = []
            for oi in range(len(us)):
                if oi == ui:
                    continue
                o = us[oi]
                if (o["end"] == u["start"] or o["start"] == u["end"]
                        or o["start"] == u["start"]
                        or o["end"] == u["end"]):
                    neigh.append(o["coverage"])
            if not neigh:
                continue
            best = neigh[0]
            for v in neigh:
                if v > best:
                    best = v
            if u["coverage"] <= tip_ratio * best:
                drop.extend(u["path"])
                tips += 1
        if not drop:
            break
        g = _drop(g, drop)

    us = unitigs(g)
    if min_length is None:
        keep = us
        short = []
    else:
        keep = [u for u in us if u["length"] >= int(min_length)]
        short = [u for u in us if u["length"] < int(min_length)]
    lens = [u["length"] for u in keep]
    total = 0
    for v in lens:
        total += v
    return RichResult(payload={
        "contigs": [u["seq"] for u in keep],
        "lengths": lens,
        "coverage": [u["coverage"] for u in keep],
        "starts": [u["start"] for u in keep],
        "ends": [u["end"] for u in keep],
        "short_contigs": [u["seq"] for u in short],
        "n_contigs": len(keep),
        "n_short": len(short),
        "total_length": total,
        "longest": lens[0] if lens else 0,
        "n50": n50(lens),
        "n_tips_removed": tips,
        "n_bubbles_removed": bubbles,
        "n_kmers": len(g["edges"]),
        "n_kmers_initial": n_edges0,
        "n_nodes": len(g["nodes"]),
        "n_reads": len(rs),
        "n_reads_used": g["used"],
        "n_reads_too_short": g["short"],
        "k": k,
        "tip_length": int(tip_length),
        "tip_ratio": float(tip_ratio),
        "bubble_ratio": float(bubble_ratio),
        "method": "de Bruijn assembly with relative tip and bubble "
                  "removal",
    })


def cheatsheet():
    return ("metsem: metagenome assembly. de Bruijn graph, maximal "
            "non-branching unitigs, tips and bubbles removed on "
            "RELATIVE coverage so rare organisms survive")
