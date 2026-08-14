# morie.fn -- function file (rootcoder007/morie)
r"""FolkRank: PageRank on a folksonomy, minus what PageRank always
says.

A folksonomy is not a web graph. Its data are **triples** -- user,
tag, resource -- so the natural structure is an undirected **triadic
hyperedge**, not a directed binary link, and the paper is explicit
that the PageRank weight-spreading approach cannot be applied directly
because of that difference.

**The conversion, and what it costs.** Each triple becomes three
undirected edges in a tripartite graph. Undirected means weight flows
both ways, so the stationary distribution is dominated by degree:
run plain PageRank with a topic preference vector and the top of the
list is the globally popular tags, whichever topic you asked about.
That is not a bug to tune away -- it is what a random walk on this
graph does.

**FolkRank is a difference of two runs.** Compute the ranking **with**
the preference vector and **without** it, and take

.. math:: w = w^{(1)} - w^{(0)}.

Whatever is popular regardless of the query cancels; what survives is
what the preference vector actually pulled up. ``folkrank`` returns
both components alongside the difference, so the cancellation can be
inspected rather than believed -- and the anchor checks that the
globally dominant node, which tops the undifferenced ranking, does
*not* top the differential one.

**The preference vector's mass must match**, :math:`\|w\|_1 =
\|p\|_1`; typically a few entries carry a high weight and the rest is
spread evenly.

References
----------
Hotho, A., Jaschke, R., Schmitz, C. & Stumme, G. (2006) "Information
Retrieval in Folksonomies: Search and Ranking", *The Semantic Web:
Research and Applications (ESWC 2006)*, LNCS 4011, 411-426,
doi:10.1007/11762256_31. [PDF supplied by Vee.] The formal folksonomy
model of users, tags and resources; that the PageRank weight-spreading
approach cannot be applied directly to folksonomies because of the
different nature of the structure -- undirected triadic hyperedges
instead of directed binary edges; the adaptation of PageRank to the
tripartite graph; the finding that this alone gives insufficiently
topic-specific results, motivating a more sophisticated algorithm; and
the FolkRank differential approach, comparing the rankings with and
without the preference vector, with the constraint that the preference
vector's L1 mass equals the weight vector's.

Brin, S. & Page, L. (1998) "The anatomy of a large-scale hypertextual
Web search engine", *Computer Networks and ISDN Systems* 30(1-7),
107-117, doi:10.1016/S0169-7552(98)00110-X. PageRank.

Haveliwala, T. H. (2002) "Topic-sensitive PageRank", *WWW '02*,
517-526, doi:10.1145/511446.511513. The preference-vector
personalisation being differenced.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["tripartite_graph", "adapted_pagerank", "folkrank",
           "preference_vector"]

_EPS = 1e-12


def tripartite_graph(triples):
    r"""Each (user, tag, resource) triple becomes three UNDIRECTED
    edges.

    The hyperedge is what the data actually is; this is the
    approximation the method rests on, and undirected edges are why
    degree dominates.
    """
    nodes, edges = set(), {}
    for (u, t, r) in triples:
        nu, nt, nr = ("u:%s" % u, "t:%s" % t, "r:%s" % r)
        nodes.update([nu, nt, nr])
        for a, b in ((nu, nt), (nt, nr), (nu, nr)):
            edges[(a, b)] = edges.get((a, b), 0) + 1
            edges[(b, a)] = edges.get((b, a), 0) + 1
    adj = {}
    for (a, b), w in edges.items():
        adj.setdefault(a, {})[b] = float(w)
    return {"adjacency": adj, "nodes": sorted(nodes),
            "n_nodes": len(nodes), "n_triples": len(list(triples)),
            "note": "a triadic hyperedge flattened to three "
                    "undirected edges, so weight flows both ways"}


def preference_vector(nodes, focus, weight=0.9):
    r"""Mass on the topic, the remainder spread evenly.

    :math:`\|p\|_1 = 1` here, so it can be compared with a uniform
    vector of the same mass.
    """
    N = list(nodes)
    F = [f for f in focus if f in set(N)]
    if not F:
        raise ValueError("tagRC: none of the focus nodes are in the "
                         "graph")
    w = float(weight)
    if not 0.0 < w < 1.0:
        raise ValueError("tagRC: the focus weight must lie in (0,1)")
    rest = len(N) - len(F)
    p = {}
    for n in N:
        p[n] = (w / len(F)) if n in F else \
            ((1.0 - w) / rest if rest else 0.0)
    return {"p": p, "focus": F, "mass": sum(p.values())}


def adapted_pagerank(adjacency, nodes, p=None, d=0.7, iters=200,
                     tol=1e-12):
    r"""Weight spreading on the undirected tripartite graph."""
    N = list(nodes)
    n = len(N)
    if n == 0:
        raise ValueError("tagRC: the graph is empty")
    if not 0.0 < float(d) < 1.0:
        raise ValueError("tagRC: the damping factor must lie in "
                         "(0,1)")
    pref = {u: 1.0 / n for u in N} if p is None else dict(p)
    w = {u: 1.0 / n for u in N}
    deg = {u: sum(adjacency.get(u, {}).values()) for u in N}
    for _ in range(int(iters)):
        nxt = {}
        for u in N:
            s = 0.0
            for v in N:
                a = adjacency.get(v, {}).get(u, 0.0)
                if a > 0.0 and deg[v] > _EPS:
                    s += w[v] * a / deg[v]
            nxt[u] = float(d) * s + (1.0 - float(d)) * pref.get(u,
                                                                0.0)
        tot = sum(nxt.values()) or 1.0
        nxt = {u: v / tot for u, v in nxt.items()}
        delta = max(abs(nxt[u] - w[u]) for u in N)
        w = nxt
        if delta < float(tol):
            break
    return {"w": w, "ranking": sorted(N, key=lambda u: -w[u])}


def folkrank(triples, focus, d=0.7, weight=0.9, iters=200):
    r""":math:`w = w^{(1)} - w^{(0)}`: with the preference, minus
    without.

    What is popular whatever you asked cancels; what the preference
    actually pulled up survives.
    """
    g = tripartite_graph(triples)
    N = g["nodes"]
    pv = preference_vector(N, focus, weight)
    with_p = adapted_pagerank(g["adjacency"], N, pv["p"], d, iters)
    without = adapted_pagerank(g["adjacency"], N, None, d, iters)
    diff = {u: with_p["w"][u] - without["w"][u] for u in N}
    order = sorted(N, key=lambda u: -diff[u])
    return RichResult(payload={
        "estimate": order, "ranking": order, "difference": diff,
        "with_preference": with_p["w"],
        "without_preference": without["w"],
        "undifferenced_ranking": with_p["ranking"],
        "baseline_ranking": without["ranking"],
        "focus": pv["focus"], "n_nodes": g["n_nodes"],
        "method": "FolkRank differential ranking; Hotho, Jaschke, "
                  "Schmitz & Stumme (2006)",
        "note": "PageRank cannot be applied directly -- undirected "
                "triadic hyperedges, not directed binary edges -- and "
                "on this graph degree dominates, which is what the "
                "difference removes",
    })


def cheatsheet():
    return ("tagRC: a folksonomy is (user, tag, resource) TRIPLES, so "
            "the structure is an undirected triadic HYPEREDGE, not a "
            "directed binary link -- PageRank cannot be applied "
            "directly. Flatten each triple to three undirected edges; "
            "because they are undirected, DEGREE dominates the "
            "stationary distribution and a topic preference vector "
            "still returns the globally popular nodes. FOLKRANK is the "
            "difference of two runs, WITH and WITHOUT the preference "
            "vector: whatever is popular regardless of the query "
            "cancels, and what the preference actually pulled up "
            "survives. Keep ||p||_1 = ||w||_1.")


# compact alias per ledger/NAMING.md
folkrank_search = folkrank
