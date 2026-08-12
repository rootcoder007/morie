r"""De novo fragment assembly by the Eulerian-path (de Bruijn) approach.

Pevzner, P. A., Tang, H., & Waterman, M. S. (2001) "An Eulerian path
approach to DNA fragment assembly", *PNAS* 98(17), 9748-9753.

The paper abandons the twenty-year-old "overlap-layout-consensus"
paradigm. The reason is stated sharply: laying reads out directly
leads to the **NP-complete Hamiltonian Path Problem** -- visit every
*read* once -- whereas breaking each read of length :math:`n` into its
:math:`n - l + 1` overlapping :math:`l`-tuples (Idury & Waterman)
turns the same task into the **Eulerian Path Problem** -- visit every
*edge* once -- which is easy. Some read-level information is lost, but
"the loss of information is minimal for large :math:`l` and is well
paid for by the computational advantages".

The **de Bruijn graph** :math:`G(S, l)` of a read set
:math:`S = \{s_1, \dots, s_n\}`: the vertices are the
:math:`(l-1)`-tuples occurring in :math:`S`, and an edge joins
:math:`v` to :math:`w` whenever :math:`S` contains an :math:`l`-tuple
whose first :math:`l-1` nucleotides are :math:`v` and whose last
:math:`l-1` are :math:`w`. Every :math:`l`-tuple is one edge. A
sequence that generated the reads corresponds to a path using every
edge -- a Chinese Postman path -- and introducing edge multiplicities
turns that into an Eulerian path.

Implemented here:

* :func:`de_bruijn_graph` -- the construction above, with
  multiplicities;
* :func:`eulerian_path` -- Hierholzer's algorithm, plus the exact
  existence condition for a directed multigraph (connected on its
  non-isolated vertices, and either all in-degrees equal out-degrees,
  or exactly one vertex with :math:`\text{out} - \text{in} = 1` and one
  with :math:`\text{in} - \text{out} = 1`);
* :func:`asmnvr` -- assemble reads into contigs, returning the full
  sequence when an Eulerian path exists and unambiguous *unitigs*
  (maximal non-branching paths) when it does not.

Where this stops, and why that is the honest boundary: the paper's
contribution beyond the graph is **error correction** and the
**Eulerian superpath** machinery for using reads to resolve repeats,
because sequencing errors "transform a simple de Bruijn graph into a
tangle of erroneous edges" and repeats create many Eulerian paths, only
one of which is the genome. Those are not implemented, so this module
does not claim to resolve repeats -- when the graph branches it reports
unitigs and says the assembly is ambiguous, rather than picking an
arbitrary Eulerian path and presenting it as the answer.
"""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["asmnvr", "de_bruijn_graph", "eulerian_path",
           "de_novo_assembly"]


def de_bruijn_graph(reads, k, multiplicity="set"):
    r"""Build :math:`G(S, l)` with :math:`l = k` (Pevzner et al. 2001).

    Vertices are :math:`(k-1)`-mers; each :math:`k`-mer is an edge from
    its prefix to its suffix.

    ``multiplicity`` decides what a repeated :math:`k`-mer means, and
    the distinction matters:

    ``"set"`` (default)
        One edge per *distinct* :math:`k`-mer, which is the
        Idury-Waterman / SBH construction the paper builds on. Reads
        overlap heavily, so a :math:`k`-mer typically occurs in many
        reads; that is **coverage**, not genome copy number, and
        counting it would give a graph whose Eulerian paths traverse
        each locus once per read.
    ``"count"``
        One edge per occurrence. The paper does use multiplicities --
        "one can substitute every edge in the de Bruijn graph by
        :math:`k` parallel edges, where :math:`k` is the number of
        times the edge is used in the Chinese Postman path" -- but
        that :math:`k` is the number of times the *genome* traverses
        the edge. Recovering it from coverage is an inference problem
        this module does not attempt, so ``"count"`` is offered for
        callers who already know their multiplicities and is not the
        default.

    Returns ``(edges, indeg, outdeg)``.
    """
    if multiplicity not in ("set", "count"):
        raise ValueError("asmnvr: multiplicity must be 'set' or 'count', "
                         "got %r" % (multiplicity,))
    k = int(k)
    if k < 2:
        raise ValueError("asmnvr: k must be >= 2 (a k-mer needs a "
                         "(k-1)-mer prefix and suffix)")
    rs = [str(r) for r in reads]
    if not rs:
        raise ValueError("asmnvr: reads must be non-empty")
    kmers = []
    seen = set()
    for r in rs:
        if len(r) < k:
            continue
        for i in range(len(r) - k + 1):
            kmer = r[i:i + k]
            if multiplicity == "set":
                if kmer in seen:
                    continue
                seen.add(kmer)
            kmers.append(kmer)

    edges = {}
    indeg = {}
    outdeg = {}
    n_kmers = 0
    if True:
        for kmer in kmers:
            v, w = kmer[:-1], kmer[1:]
            edges.setdefault(v, []).append(w)
            outdeg[v] = outdeg.get(v, 0) + 1
            indeg[w] = indeg.get(w, 0) + 1
            indeg.setdefault(v, indeg.get(v, 0))
            outdeg.setdefault(w, outdeg.get(w, 0))
            n_kmers += 1
    if n_kmers == 0:
        raise ValueError("asmnvr: no read is at least k = %d long" % k)
    return edges, indeg, outdeg


def _connected(edges, verts):
    """Weak connectivity over vertices that carry at least one edge."""
    active = [v for v in verts if edges.get(v) or
              any(v in ws for ws in edges.values())]
    if not active:
        return True
    adj = {}
    for v, ws in edges.items():
        for w in ws:
            adj.setdefault(v, set()).add(w)
            adj.setdefault(w, set()).add(v)
    start = active[0]
    seen = set([start])
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return all(v in seen for v in active)


def eulerian_path(edges, indeg, outdeg):
    r"""Hierholzer's algorithm, with the existence condition checked.

    A connected directed multigraph has an Eulerian path iff every
    vertex has equal in- and out-degree (then it is a circuit), or
    exactly one vertex has :math:`\text{out} - \text{in} = 1` (the
    start) and exactly one has :math:`\text{in} - \text{out} = 1` (the
    end). Returns ``None`` when no such path exists -- an ambiguous or
    disconnected graph is reported, not guessed at.
    """
    verts = set(list(indeg) + list(outdeg))
    starts = [v for v in verts if outdeg.get(v, 0) - indeg.get(v, 0) == 1]
    ends = [v for v in verts if indeg.get(v, 0) - outdeg.get(v, 0) == 1]
    odd = [v for v in verts
           if abs(indeg.get(v, 0) - outdeg.get(v, 0)) > 1]
    if odd or len(starts) > 1 or len(ends) > 1 or len(starts) != len(ends):
        return None
    if not _connected(edges, verts):
        return None
    if starts:
        start = starts[0]
    else:
        cand = [v for v in sorted(verts, key=repr) if edges.get(v)]
        if not cand:
            return None
        start = cand[0]

    nxt = dict((v, list(ws)) for v, ws in edges.items())
    stack = [start]
    path = []
    while stack:
        v = stack[-1]
        if nxt.get(v):
            stack.append(nxt[v].pop())
        else:
            path.append(stack.pop())
    path.reverse()
    total = sum(len(ws) for ws in edges.values())
    if len(path) != total + 1:
        return None                      # not all edges were used
    return path


def _unitigs(edges, indeg, outdeg):
    """Maximal non-branching paths: the part of the assembly that is
    unambiguous even when the whole graph is not."""
    def simple(v):
        return indeg.get(v, 0) == 1 and outdeg.get(v, 0) == 1

    out = []
    used = set()
    for v in sorted(edges, key=repr):
        if simple(v):
            continue
        for idx, w in enumerate(edges[v]):
            if (v, idx) in used:
                continue
            used.add((v, idx))
            walk = [v, w]
            cur = w
            while simple(cur) and edges.get(cur):
                nxt_v = edges[cur][0]
                used.add((cur, 0))
                walk.append(nxt_v)
                cur = nxt_v
            out.append(walk)
    # a pure cycle with no branch point contributes nothing above
    if not out and edges:
        v = sorted(edges, key=repr)[0]
        walk = [v]
        cur = v
        seen = set()
        while edges.get(cur) and (cur, 0) not in seen:
            seen.add((cur, 0))
            cur = edges[cur][0]
            walk.append(cur)
            if cur == v:
                break
        out.append(walk)
    return out


def _spell(path):
    if not path:
        return ""
    s = path[0]
    for v in path[1:]:
        s += v[-1]
    return s


def asmnvr(reads, k=None, multiplicity="set"):
    r"""Assemble reads via the de Bruijn graph and an Eulerian path.

    Parameters
    ----------
    reads : sequence of str
        The reads. They need not be the same length.
    k : int, optional
        The :math:`l` of the paper. Defaults to the shortest read
        length, which makes every read contribute at least one
        :math:`k`-mer.
    multiplicity : {"set", "count"}
        Whether a repeated :math:`k`-mer is one edge or many; see
        :func:`de_bruijn_graph`. ``"set"`` is right when repetition
        comes from read coverage, which is the usual case.

    Returns
    -------
    RichResult
        ``estimate`` / ``sequence`` is the assembled sequence when an
        Eulerian path exists, otherwise ``None``. ``contigs`` are the
        unitigs (maximal non-branching paths) spelled out, always
        available; ``unambiguous`` says whether the Eulerian path is
        unique-by-construction here (it is not, in general -- see the
        module docstring); ``n_kmers``, ``n_vertices``, ``branching``
        and ``graph`` describe the graph itself.

    Examples
    --------
    Reads tiling a sequence reassemble it exactly::

        s = "ATGGCGTGCA"
        reads = [s[i:i+5] for i in range(len(s)-4)]
        asmnvr(reads, k=4)["sequence"]      # "ATGGCGTGCA"

    References
    ----------
    Pevzner, Tang & Waterman (2001) *PNAS* 98(17), 9748-9753,
    "Eulerian Superpaths".
    """
    rs = [str(r) for r in reads]
    if not rs:
        raise ValueError("asmnvr: reads must be non-empty")
    if k is None:
        k = min(len(r) for r in rs)
    edges, indeg, outdeg = de_bruijn_graph(rs, k, multiplicity)
    path = eulerian_path(edges, indeg, outdeg)
    # With multiplicity="set" every distinct k-mer is ONE edge, so a
    # k-mer that genuinely repeats in the source is traversed once and the
    # assembly comes out short: reads tiling ATGCATGC at k = 3 assemble to
    # ATGCAT. Nothing downstream can notice, because the path really is
    # Eulerian on the graph that was built -- and the collapse is not
    # detectable from the k-mer set either, since read coverage produces
    # repeated k-mers too. That is the paper's own point about repeats,
    # and the reason its Eulerian-superpath machinery exists. So the
    # length is reported as what it is, a lower bound, rather than
    # guessed at.
    lower_bound = (multiplicity == "set")
    contigs = [_spell(p) for p in _unitigs(edges, indeg, outdeg)]
    branching = sorted(
        [v for v in set(list(indeg) + list(outdeg))
         if outdeg.get(v, 0) > 1 or indeg.get(v, 0) > 1], key=repr)
    n_kmers = sum(len(ws) for ws in edges.values())
    return RichResult(payload={
        "estimate": _spell(path) if path else None,
        "sequence": _spell(path) if path else None,
        "path": path,
        "contigs": contigs,
        "unambiguous": bool(path is not None and not branching),
        "length_is_lower_bound": bool(lower_bound and path is not None),
        "branching": branching,
        "n_kmers": n_kmers,
        "n_vertices": len(set(list(indeg) + list(outdeg))),
        "graph": edges,
        "k": int(k),
        "multiplicity": multiplicity,
        "note": "repeat resolution (Eulerian superpaths) and error "
                "correction are NOT implemented; a branching graph is "
                "reported as ambiguous rather than resolved. With "
                "multiplicity='set' a k-mer that repeats in the source "
                "is one edge and is traversed once, so the assembled "
                "length is a LOWER BOUND on the truth "
                "(length_is_lower_bound); the collapse cannot be "
                "detected from the k-mer set, since read coverage "
                "repeats k-mers too",
        "method": "Eulerian-path assembly (Pevzner, Tang & Waterman 2001)",
    })


def cheatsheet():
    return ("asmnvr: de Bruijn assembly (Pevzner 2001). Break reads "
            "into l-tuples; vertices are (l-1)-tuples, each l-tuple is "
            "an EDGE. Overlap-layout-consensus needs a Hamiltonian "
            "path (NP-complete); this needs an EULERIAN path (easy). "
            "Hierholzer, with the exact existence condition. Repeat "
            "resolution via Eulerian superpaths and error correction "
            "are NOT here -- branching graphs are reported ambiguous "
            "and only unitigs are claimed.")


# compact alias per ledger/NAMING.md
de_novo_assembly = asmnvr
