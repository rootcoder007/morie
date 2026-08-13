# morie.fn -- shelf core (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Causal-inference / causal-discovery shelf core.

Spec: Molak, A., *Causal Inference and Discovery in Python*, Packt.
The copy in the corpus carries the 2023 first-edition copyright page,
so every locator below is a first-edition page; the backlog labels the
shelf as the 2025 second edition.  Where a construct is only *named*
in that copy and no formula is printed, the docstring says so in those
words rather than attributing a formula to the book.

Graph work reuses the path machinery already in ``bdcrt`` and the
d-separation query in ``_dsep``; HSIC reuses ``anmod.hsic``.  Nothing
here draws random numbers.
"""

from __future__ import annotations

import math

from . import _array_core as np
from ._dsep import d_separated
from .anmod import hsic as _hsic
from .bdcrt import _descendants, _has_cycle, _parse, _paths

_PARENTS = 0


def _edgelist(dag):
    """Normalize any accepted DAG spelling to a sorted list of (u, v)."""
    children, _parents, _nodes = _parse(dag)
    out = sorted((u, v) for u, vs in children.items() for v in vs)
    return out


def _nodes_of(edges, extra=()):
    s = set()
    for u, v in edges:
        s.add(u)
        s.add(v)
    s.update(extra)
    return sorted(s)


def _dag(edges, nodes):
    """Adjacency dict over a FIXED node set, so cutting edges never
    silently deletes an isolated node from the graph."""
    d = {n: [] for n in nodes}
    for u, v in edges:
        d.setdefault(u, []).append(v)
        d.setdefault(v, [])
    return {k: sorted(v) for k, v in d.items()}


def _cut_incoming(edges, targets):
    t = set(targets)
    return [(u, v) for u, v in edges if v not in t]


def _cut_outgoing(edges, sources):
    s = set(sources)
    return [(u, v) for u, v in edges if u not in s]


def _skeleton(edges):
    return sorted({tuple(sorted(e)) for e in edges})


def _adjacent(edges, a, b):
    return (a, b) in edges or (b, a) in edges


def _colliders(edges):
    """Every immorality a -> c <- b with a, b non-adjacent, sorted."""
    edges = list(edges)
    pars = {}
    for u, v in edges:
        pars.setdefault(v, set()).add(u)
    out = []
    for c in sorted(pars):
        ps = sorted(pars[c])
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                if not _adjacent(edges, ps[i], ps[j]):
                    out.append((ps[i], c, ps[j]))
    return sorted(out)


# --- ch. 13, p. 348: GES scoring --------------------------------------

def bicdag(data, dag, names=None):
    """Gaussian BIC score of a DAG.

    The corpus copy only NAMES the Bayesian Information Criterion as a
    gCastle GES scoring option (ch. 13, p. 348, citing Chickering
    2003); no BIC formula is printed there.  The formula used here is
    the standard Gaussian one, stated explicitly so nothing is
    attributed to the book that the book does not say:

        score(G) = sum_j [ -n/2 (log(2 pi s2_j) + 1) ] - (log n / 2) k

    with ``s2_j`` the residual variance of the OLS regression of node j
    on its parents and ``k`` the free-parameter count (one intercept,
    one slope per parent, one variance per node).
    """
    rows = [[float(v) for v in r] for r in data]
    n = len(rows)
    if n < 2:
        raise ValueError("need at least 2 rows of data")
    p = len(rows[0])
    if any(len(r) != p for r in rows):
        raise ValueError("all data rows must have the same length")
    names = list(names) if names is not None else list(range(p))
    if len(names) != p:
        raise ValueError("names must have one entry per column")
    idx = {nm: i for i, nm in enumerate(names)}
    edges = _edgelist(dag)
    pars = {nm: [] for nm in names}
    for u, v in edges:
        if u not in idx or v not in idx:
            raise ValueError("DAG node %r has no data column" % (u if u not in idx else v,))
        pars[v].append(u)
    total = 0.0
    k = 0
    for nm in names:
        y = [r[idx[nm]] for r in rows]
        ps = sorted(pars[nm])
        design = [[1.0] + [r[idx[q]] for q in ps] for r in rows]
        xm = np.asarray(design, dtype=float)
        beta = np.dot(np._pinv(xm), np.asarray(y, dtype=float))
        fit = np.dot(xm, beta)
        rss = sum((y[i] - float(fit[i])) ** 2 for i in range(n))
        s2 = max(rss / n, 1e-300)
        total += -0.5 * n * (math.log(2.0 * math.pi * s2) + 1.0)
        k += len(ps) + 2
    return {
        "score": total - 0.5 * math.log(n) * k,
        "loglik": total,
        "k": k,
        "penalty": 0.5 * math.log(n) * k,
        "n": n,
    }


# --- bow arcs ---------------------------------------------------------

def bowarc(dag, bidirected, x, y):
    """Whether the pair (x, y) forms a bow.

    NOT LOCATED IN THE EXTRACTED TEXT of the corpus copy of Molak,
    which contains no discussion of bows or bow-free graphs.  The
    definition and the theorem are therefore taken from the primary
    source and quoted from it:

        "A bow-arc is a pair of variables, one of which is a direct
        function of the other, whose error terms are correlated."

        "Theorem 4. (Brito and Pearl, 2002b) (Bow-free Rule) Every
        acyclic model whose path diagram lacks bow-arcs is identified."

    -- Chen, B. and Pearl, J. (2015), *Graphical Tools for Linear
    Structural Equation Modeling*, UCLA Cognitive Systems Laboratory
    Technical Report R-432, p. 15, citing Brito, C. and Pearl, J.
    (2002), "A new identification condition for recursive models with
    correlated errors", Structural Equation Modeling 9(4):459-474.

    Footnote 17 of R-432 gives the equivalent phrasing used for
    ``bowfree`` here: "a bow-free model is a model where error terms of
    every parent-child pair are not correlated".

    ``identified`` reports Theorem 4 itself: acyclic AND bow-free.  The
    theorem addresses whole-model identifiability, not the
    identifiability of individual coefficients in unidentified models,
    so nothing stronger is claimed.
    """
    edges = _edgelist(dag)
    children, _parents, nodes = _parse(dag)
    acyclic = not _has_cycle(children, nodes)
    bid = sorted({tuple(sorted(e)) for e in bidirected})
    direct = (x, y) in edges
    confounded = tuple(sorted((x, y))) in bid
    allbows = sorted(
        (u, v) for (u, v) in edges if tuple(sorted((u, v))) in set(bid)
    )
    return {
        "isbow": bool(direct and confounded),
        "direct": bool(direct),
        "confounded": bool(confounded),
        "nbows": len(allbows),
        "bowfree": len(allbows) == 0,
        "acyclic": bool(acyclic),
        "identified": bool(acyclic and len(allbows) == 0),
        "bows": allbows,
    }


# --- ch. 5, pp. 82-85: colliders and Markov equivalence ---------------

def collider(dag, triple=None):
    """Collider structures (immoralities, v-structures), ch. 5 p. 82.

    With ``triple=(a, c, b)`` the return also says whether that one
    triple is a collider at ``c``.
    """
    edges = _edgelist(dag)
    cols = _colliders(edges)
    hit = False
    shielded = False
    if triple is not None:
        a, c, b = triple
        hit = (a, c, b) in cols or (b, c, a) in cols
        shielded = (a, c) in edges and (b, c) in edges and _adjacent(edges, a, b)
    return {
        "ncolliders": len(cols),
        "colliders": cols,
        "iscollider": bool(hit),
        "shielded": bool(shielded),
        "nedges": len(edges),
    }


def mectest(dag1, dag2):
    """Markov equivalence, ch. 5 p. 85 (Verma and Pearl, 1991).

    Two DAGs are Markov equivalent iff they share a skeleton and a set
    of colliders.
    """
    e1 = _edgelist(dag1)
    e2 = _edgelist(dag2)
    s1, s2 = _skeleton(e1), _skeleton(e2)
    c1, c2 = _colliders(e1), _colliders(e2)
    return {
        "equivalent": bool(s1 == s2 and c1 == c2),
        "sameskeleton": bool(s1 == s2),
        "samecolliders": bool(c1 == c2),
        "nskeleton": len(s1),
        "ncolliders1": len(c1),
        "ncolliders2": len(c2),
    }


# --- ch. 6, p. 119: the three rules of do-calculus ---------------------

def docalc(dag, y, z, x=(), w=()):
    """Which of the three rules of do-calculus applies, p. 119.

    Notation as printed: ``G`` with an overline on X drops every edge
    INTO X; with an underline on Z drops every edge OUT OF Z.

    * Rule 1 (ignore an observation): (Y indep Z | X, W) in G_Xbar
    * Rule 2 (treat an intervention as an observation):
      (Y indep Z | X, W) in G_Xbar,Zunder
    * Rule 3 (ignore an intervention): (Y indep Z | X, W) in
      G_Xbar,Zbar(W), where Z(W) is the subset of Z that are not
      ancestors of any W node in G_Xbar.
    """
    edges = _edgelist(dag)
    x = tuple(x)
    w = tuple(w)
    zs = (z,) if isinstance(z, str) else tuple(z)
    if len(zs) != 1:
        raise ValueError("this implementation checks one z node at a time")
    zn = zs[0]
    cond = sorted(set(x) | set(w))
    allnodes = _nodes_of(edges, (y, zn) + x + w)

    g1 = _cut_incoming(edges, x)
    r1 = d_separated(_dag(g1, allnodes), y, zn, cond)

    g2 = _cut_outgoing(_cut_incoming(edges, x), zs)
    r2 = d_separated(_dag(g2, allnodes), y, zn, cond)

    # Z(W): the z nodes that are NOT ancestors of any w node in G_Xbar.
    ch1 = _dag(g1, allnodes)
    anc_of_w = set()
    for node in allnodes:
        if _descendants(node, ch1) & set(w):
            anc_of_w.add(node)
    zw = tuple(q for q in zs if q not in anc_of_w)
    g3 = _cut_incoming(_cut_incoming(edges, x), zw)
    r3 = d_separated(_dag(g3, allnodes), y, zn, cond)

    return {
        "rule1": bool(r1),
        "rule2": bool(r2),
        "rule3": bool(r3),
        "nrules": int(r1) + int(r2) + int(r3),
        "zwsize": len(zw),
    }


def dointerv(dag, x):
    """The do-operator as graph surgery (modularity, ch. 7 p. 154).

    ``do(X = x)`` deletes every edge into X and leaves every other
    structural equation untouched.
    """
    edges = _edgelist(dag)
    xs = (x,) if isinstance(x, str) else tuple(x)
    kept = _cut_incoming(edges, xs)
    removed = [e for e in edges if e not in kept]
    return {
        "edges": kept,
        "removed": removed,
        "nremoved": len(removed),
        "nkept": len(kept),
        "nnodes": len(_nodes_of(edges, xs)),
    }


# --- ch. 6: d-separation ----------------------------------------------

def dseptest(dag, x, y, z=()):
    """d-separation of x and y given z, ch. 6.

    Reuses ``morie.fn._dsep.d_separated``; the extra counts come from
    the same path enumeration so callers can see WHY the answer came
    out the way it did.
    """
    edges = _edgelist(dag)
    children, parents, nodes = _parse(edges)
    paths = _paths(x, y, children, parents)
    sep = d_separated(edges, x, y, tuple(z))
    return {
        "dseparated": bool(sep),
        "npaths": len(paths),
        "nnodes": len(nodes),
        "ncond": len(tuple(z)),
    }


# --- ch. 5, p. 77: the faithfulness assumption -------------------------

def faithchk(dag, x, y, z=(), indep=True):
    """Faithfulness for one triple, ch. 5 p. 77.

    The printed formulation is ``X indep_P Y | Z  =>  X indep_G Y | Z``:
    an independence in the DISTRIBUTION must be reflected in the GRAPH.
    ``indep`` is the observed distributional independence; the return
    says whether that implication survives for this triple, and
    separately whether the converse (the global Markov property) does.
    """
    sep = d_separated(_edgelist(dag), x, y, tuple(z))
    indep = bool(indep)
    return {
        "dseparated": bool(sep),
        "indep": indep,
        "faithful": bool((not indep) or sep),
        "markov": bool((not sep) or indep),
        "violation": bool(indep and not sep),
    }


# --- ch. 13, p. 354: HSIC ---------------------------------------------

def hsicstat(a, b, sigma_a=None, sigma_b=None, threshold=0.01):
    """Hilbert-Schmidt independence criterion for an ANM residual test.

    The corpus copy only CALLS gCastle's ``hsic_test`` (ch. 13, p. 354)
    and prints no formula; the book's own citation for the criterion is
    Gretton et al. (2007).  The estimator here is the existing biased
    V-statistic in ``morie.fn.anmod.hsic``, ``tr(K H L H)/n^2`` with
    RBF Gram matrices and the median-heuristic bandwidth, reused rather
    than reimplemented.  ``threshold`` is a caller-supplied cutoff --
    no null distribution is simulated, so nothing here is random.
    """
    stat = float(_hsic(a, b, sigma_a, sigma_b))
    n = len(list(a))
    return {
        "hsic": stat,
        "nhsic": stat * n,
        "independent": bool(stat < float(threshold)),
        "n": n,
    }


# --- ch. 2, p. 15: the Ladder of Causation -----------------------------

_RUNGS = {
    1: ("Association", "Observing", "How does observing X change my belief in Y?"),
    2: ("Intervention", "Doing", "What will happen to Y if I do X?"),
    3: ("Counterfactual", "Imagining", "If I had done X, what would Y be?"),
}


def causrung(rung):
    """One rung of the Ladder of Causation, Table 2.1, ch. 2 p. 15.

    The three rungs and their actions and questions are transcribed
    from the printed table.  ``needsgraph`` and ``needsscm`` follow the
    book's own account of what each rung requires: rung 1 needs only a
    joint distribution, rung 2 needs a causal graph, rung 3 needs a
    full structural causal model.
    """
    rung = int(rung)
    if rung not in _RUNGS:
        raise ValueError("rung must be 1, 2 or 3, got %r" % (rung,))
    name, action, question = _RUNGS[rung]
    return {
        "level": rung,
        "name": name,
        "action": action,
        "question": question,
        "needsgraph": bool(rung >= 2),
        "needsscm": bool(rung >= 3),
    }


# --- ch. 7, p. 157: the positivity assumption --------------------------

def poschk(treat, stratum, tol=0.0):
    """Positivity: every treatment value has positive probability in
    every covariate stratum, ch. 7 p. 157.

    Returns the smallest stratum-conditional treatment probability over
    all (stratum, treatment level) cells, so the caller can see how
    close to a violation the data sit rather than only that it passed.
    """
    treat = list(treat)
    stratum = list(stratum)
    if len(treat) != len(stratum) or not treat:
        raise ValueError("treat and stratum must be non-empty and equal length")
    levels = sorted({t for t in treat})
    strata = sorted({s for s in stratum})
    cells = []
    for s in strata:
        idx = [i for i in range(len(treat)) if stratum[i] == s]
        for lv in levels:
            cells.append(sum(1 for i in idx if treat[i] == lv) / float(len(idx)))
    mn = min(cells)
    return {
        "minprob": mn,
        "holds": bool(mn > float(tol)),
        "ncells": len(cells),
        "nstrata": len(strata),
        "nlevels": len(levels),
    }


# --- the R-learner ----------------------------------------------------

def rlearn(y, t, m, e, x=None):
    """Residualized (Robinson-style) CATE estimator.

    The decomposition is Robinson, P. M. (1988),
    "Root-N-Consistent Semiparametric Regression", Econometrica
    56(4), 931-954. PDF not in hand (JSTOR serves HTML); cited
    from bibliographic details. Nie and Wager (2021), cited
    below, is the R-learner built on it and is the spec actually
    followed here.

    NOT LOCATED IN THE EXTRACTED TEXT of the corpus copy of Molak,
    which covers the S-, T-, X- and DR-learners but has no R-learner
    section.  The estimator is therefore taken from the primary source
    and quoted from it:

        Robinson decomposition, eq. (1):
            "Y_i - m*(X_i) = {W_i - e*(X_i)} tau*(X_i) + eps_i"

        R-learner objective, eq. (4):
            "tau_hat(.) = argmin_tau [ L_hat_n{tau(.)} + Lambda_n{tau(.)} ]"
        with
            "L_hat_n{tau(.)} = (1/n) sum_i [ {Y_i - m_hat^(-q(i))(X_i)}
                 - {W_i - e_hat^(-q(i))(X_i)} tau(X_i) ]^2"

    -- Nie, X. and Wager, S. (2021), "Quasi-Oracle Estimation of
    Heterogeneous Treatment Effects", Biometrika 108(2):299-319
    (arXiv:1712.04912), where "e*(x) = pr(W = 1 | X = x)" and
    "m*(x) = E(Y | X = x)".

    This routine computes eq. (4) with the regularizer Lambda_n set to
    zero and the CROSS-FITTED nuisance predictions m_hat^(-q(i)) and
    e_hat^(-q(i)) supplied BY THE CALLER, so the fold assignment q(.)
    -- the only random ingredient in the paper's construction -- lives
    outside this function and both language arms see identical numbers.
    ``x`` gives basis columns for a linear tau(X); omit it for a
    constant treatment effect.
    """
    y = [float(v) for v in y]
    t = [float(v) for v in t]
    m = [float(v) for v in m]
    e = [float(v) for v in e]
    n = len(y)
    if not n or any(len(v) != n for v in (t, m, e)):
        raise ValueError("y, t, m, e must be non-empty and the same length")
    ry = [y[i] - m[i] for i in range(n)]
    rt = [t[i] - e[i] for i in range(n)]
    if x is None:
        num = sum(rt[i] * ry[i] for i in range(n))
        den = sum(rt[i] * rt[i] for i in range(n))
        if den <= 0.0:
            raise ValueError("residualized treatment has no variation")
        tau = [num / den]
        pred = [tau[0]] * n
    else:
        basis = [[1.0] + [float(v) for v in row] for row in x]
        if len(basis) != n:
            raise ValueError("x must have one row per observation")
        design = [[rt[i] * c for c in basis[i]] for i in range(n)]
        xm = np.asarray(design, dtype=float)
        coef = np.dot(np._pinv(xm), np.asarray(ry, dtype=float))
        tau = [float(v) for v in coef]
        pred = [sum(tau[j] * basis[i][j] for j in range(len(tau))) for i in range(n)]
    loss = sum((ry[i] - rt[i] * pred[i]) ** 2 for i in range(n))
    return {
        "tau": tau,
        "ate": sum(pred) / n,
        "loss": loss,
        "n": n,
        "k": len(tau),
    }


# --- separating sets ---------------------------------------------------

def sepset(dag, x, y, maxsize=3):
    """Smallest set that d-separates x from y, searched in a fixed order.

    The corpus copy discusses conditioning sets that block paths (ch. 6)
    but prints no named "separating set" definition, so the search rule
    is stated here: candidate sets are drawn from the union of the
    adjacencies of x and y, taken in sorted order and in increasing
    size, and the FIRST separating set found is returned.  That order
    is deterministic, so both language arms return the same set.
    """
    edges = _edgelist(dag)
    nodes = _nodes_of(edges)
    adj = sorted(
        {v for u, v in edges if u in (x, y)} | {u for u, v in edges if v in (x, y)}
    )
    cand = [q for q in adj if q not in (x, y)]
    if _adjacent(edges, x, y):
        return {"found": False, "size": -1, "sepset": [], "ntested": 0, "nnodes": len(nodes)}
    tested = 0
    for size in range(0, min(int(maxsize), len(cand)) + 1):
        for comb in _combinations(cand, size):
            tested += 1
            if d_separated(edges, x, y, comb):
                return {
                    "found": True,
                    "size": size,
                    "sepset": list(comb),
                    "ntested": tested,
                    "nnodes": len(nodes),
                }
    return {"found": False, "size": -1, "sepset": [], "ntested": tested, "nnodes": len(nodes)}


def _combinations(seq, k):
    if k == 0:
        yield ()
        return
    for i in range(len(seq) - k + 1):
        for rest in _combinations(seq[i + 1:], k - 1):
            yield (seq[i],) + rest


# --- ch. 7, p. 164: SUTVA ----------------------------------------------

def sutvachk(interference, versions=1, tol=0.0):
    """SUTVA: no interference between units, one version of treatment.

    The printed statement (ch. 7 p. 164) is that "the fact that one
    unit receives treatment does not influence any other units".
    ``interference`` is a square matrix whose off-diagonal entry (i, j)
    is how much unit i's treatment moves unit j's outcome; SUTVA holds
    when every off-diagonal entry is at most ``tol`` in magnitude and
    there is a single version of the treatment.
    """
    mat = [[float(v) for v in row] for row in interference]
    n = len(mat)
    if not n or any(len(r) != n for r in mat):
        raise ValueError("interference must be a square matrix")
    off = [abs(mat[i][j]) for i in range(n) for j in range(n) if i != j]
    mx = max(off) if off else 0.0
    versions = int(versions)
    return {
        "maxinterference": mx,
        "nointerference": bool(mx <= float(tol)),
        "consistent": bool(versions == 1),
        "holds": bool(mx <= float(tol) and versions == 1),
        "n": n,
    }
