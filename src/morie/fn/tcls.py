r"""t-closeness via the Earth Mover's Distance.

Li, N., Li, T., & Venkatasubramanian, S. (2007) "t-Closeness: privacy
beyond k-anonymity and l-diversity", *Proc. 23rd IEEE International
Conference on Data Engineering (ICDE)*, 106-115.

An equivalence class satisfies t-closeness when the distance between the
distribution of the sensitive attribute *within the class* (P) and its
distribution *over the whole table* (Q) is at most t; the table
satisfies t-closeness when every class does (Definition 3).

The point of the EMD, rather than variational distance or KL, is that it
respects distances *between sensitive values*. Being told someone's
salary lies in {3k, 4k, 5k} is a bigger disclosure than being told it
lies in {6k, 8k, 11k}, even though both are three values out of nine --
and only a metric that knows 3k is near 4k can see that.

Routes
------
The paper gives three ground distances, each with a closed form, so all
three are implemented and `ground` selects between them.

``"equal"`` (Sec. 5.2, categorical, no order)
    Every pair of values is at distance 1, giving
    :math:`D[P,Q] = \frac12 \sum_i |p_i - q_i|`.

``"ordered"`` (Sec. 5.1, numerical)
    With the domain sorted, :math:`d(v_i,v_j) = |i-j|/(m-1)`. Only
    flows between adjacent elements matter, which collapses to

    .. math:: D[P,Q] = \frac{1}{m-1}\sum_{i=1}^{m}
              \Big| \sum_{j \le i} (p_j - q_j) \Big|.

``"hierarchical"`` (Sec. 5.2, categorical with a taxonomy)
    :math:`d(v_1,v_2) = \mathrm{level}(v_1,v_2)/H`. Define
    :math:`\mathrm{extra}(N)` as :math:`p_i - q_i` at a leaf and the sum
    over children at an internal node; then

    .. math:: \mathrm{cost}(N) = \frac{\mathrm{height}(N)}{H}
              \min\big(\mathrm{pos\_extra}(N), \mathrm{neg\_extra}(N)\big),
              \qquad D[P,Q] = \sum_N \mathrm{cost}(N)

    over non-leaf nodes.

NOTE on the hierarchical route. It implements Sec. 5.2's printed
recursion exactly, but it does NOT reproduce the two hierarchical
numbers the paper states, and the discrepancy is the paper's:

    class                                code / hand   paper
    {gastric ulcer, gastritis,
     stomach cancer}      (Sec. 3)       0.4444        0.5
    {gastric ulcer, stomach cancer,
     pneumonia}           (Table 5)      0.2593        0.278

Both were derived by hand from the printed formulas and the code
matches the hand derivation to 1e-12. The first case settles it without
appealing to any subtlety in the recursion: under Fig. 1 the class puts
4/9 of its mass on the digestive side that has to cross to the
respiratory side, and the root ground distance is exactly 1
(level(v1,v2)/H = 3/3, as the paper's own "Flu to Stomach cancer is
3/3 = 1" states). The work is therefore 4/9 = 0.4444 and no correct EMD
can return 0.5.

So this route is trusted, and the paper's worked hierarchical values are
not. The ordered route is independently anchored on three values the
paper prints and reproduces all three exactly, which is what makes the
contrast meaningful.

The paper's own worked numbers are the anchors: with
Q = {3k,...,11k} uniform, P1 = {3k,4k,5k} gives 0.375 and
P2 = {6k,8k,11k} gives 0.167 under ``"ordered"``; the two Disease
classes give 0.5 and 0.278 under ``"hierarchical"``.
"""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["t_closeness", "emd_distance", "tcls"]

_GROUNDS = ("equal", "ordered", "hierarchical")


def _dist_over(labels, domain):
    """Empirical distribution of `labels` over the ordered `domain`."""
    n = len(labels)
    if n == 0:
        raise ValueError("t_closeness: empty group of records")
    idx = {v: i for i, v in enumerate(domain)}
    p = [0.0] * len(domain)
    for v in labels:
        if v not in idx:
            raise ValueError(
                "t_closeness: sensitive value %r is absent from the "
                "table-wide domain" % (v,))
        p[idx[v]] += 1.0 / n
    return p


def _emd_equal(p, q):
    """Sec. 5.2 equal ground distance: half the variational distance."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def _emd_ordered(p, q):
    """Sec. 5.1 ordered ground distance, via adjacent flows."""
    m = len(p)
    if m < 2:
        return 0.0
    run = 0.0
    tot = 0.0
    for a, b in zip(p, q):
        run += a - b
        tot += abs(run)
    return tot / (m - 1.0)


def _emd_hierarchical(p, q, hierarchy, domain):
    """Sec. 5.2 hierarchical ground distance.

    `hierarchy` maps each internal node to its list of children; a child
    is either another internal node or a domain value (a leaf). The root
    is the node that is nobody's child.
    """
    idx = {v: i for i, v in enumerate(domain)}
    children = {k: list(v) for k, v in hierarchy.items()}
    named = set(children)
    is_child = {c for kids in children.values() for c in kids}
    roots = [n for n in named if n not in is_child]
    if len(roots) != 1:
        raise ValueError(
            "t_closeness: hierarchy must have exactly one root, found %d"
            % len(roots))

    # Height of a node = edges down to a leaf; H = height of the root.
    def height(n):
        if n not in children:
            return 0
        return 1 + max(height(c) for c in children[n])

    H = height(roots[0])
    if H == 0:
        raise ValueError("t_closeness: hierarchy has no internal nodes")

    cost = [0.0]

    def extra(n):
        if n not in children:
            if n not in idx:
                raise ValueError(
                    "t_closeness: hierarchy leaf %r is not a domain "
                    "value" % (n,))
            return p[idx[n]] - q[idx[n]]
        kids = [extra(c) for c in children[n]]
        pos = sum(e for e in kids if e > 0)
        neg = -sum(e for e in kids if e < 0)
        # height(n)/H * min(pos_extra, neg_extra) -- the mass that has to
        # be evened out *among* n's children, charged at n's level. Mass
        # leaving the subtree entirely is charged at n's parent instead,
        # which is why this is min() and not the total.
        cost[0] += (height(n) / float(H)) * min(pos, neg)
        return sum(kids)

    extra(roots[0])
    return cost[0]


def emd_distance(p, q, ground="ordered", hierarchy=None, domain=None):
    """Earth Mover's Distance between two distributions over one domain."""
    g = str(ground).lower()
    if g not in _GROUNDS:
        raise ValueError(
            "emd_distance: ground must be one of %s, got %r"
            % (", ".join(_GROUNDS), ground))
    pv = [float(v) for v in p]
    qv = [float(v) for v in q]
    if len(pv) != len(qv):
        raise ValueError(
            "emd_distance: P has %d cells but Q has %d" % (len(pv), len(qv)))
    if g == "equal":
        return _emd_equal(pv, qv)
    if g == "ordered":
        return _emd_ordered(pv, qv)
    if hierarchy is None or domain is None:
        raise ValueError(
            "emd_distance: ground='hierarchical' needs both hierarchy "
            "and domain")
    return _emd_hierarchical(pv, qv, hierarchy, domain)


def t_closeness(X, quasi_ids, sensitive, t, ground="ordered",
                hierarchy=None, domain=None):
    r"""Check t-closeness of a release, per class and overall.

    Parameters
    ----------
    X : array-like
        Records; only the length is used, to check it against the blocks.
    quasi_ids : 2-D array-like
        Quasi-identifier tuple per record; records sharing one tuple form
        an equivalence class.
    sensitive : array-like
        Sensitive value per record; compared as labels, so strings and
        numbers both work.
    t : float
        The threshold.
    ground : {"ordered", "equal", "hierarchical"}
        Ground distance; see the module docstring.
    hierarchy : dict, optional
        node -> list of children, for ``ground="hierarchical"``.
    domain : sequence, optional
        The sensitive domain in order. Required for the ordered ground
        distance to mean anything; defaults to the sorted distinct
        values actually present.

    Returns
    -------
    RichResult
        ``estimate`` is the worst (largest) class distance -- the
        smallest t the release actually achieves; ``satisfies`` says
        whether that is within ``t``.
    """
    n = len(list(X))
    qs = [tuple(r) if isinstance(r, (list, tuple)) else (r,)
          for r in quasi_ids]
    sv = list(sensitive)
    if len(qs) != n or len(sv) != n:
        raise ValueError(
            "t_closeness: X, quasi_ids and sensitive must agree in "
            "length, got %d, %d and %d" % (n, len(qs), len(sv)))
    t = float(t)
    if t < 0.0:
        raise ValueError("t_closeness: t must be non-negative, got %r" % (t,))

    dom = (list(domain) if domain is not None
           else sorted(set(sv), key=lambda v: (str(type(v)), v)))
    q = _dist_over(sv, dom)

    blocks = {}
    for i in range(n):
        blocks.setdefault(qs[i], []).append(sv[i])
    keys = sorted(blocks, key=lambda k: tuple(str(v) for v in k))

    dists = []
    sizes = []
    for k in keys:
        p = _dist_over(blocks[k], dom)
        dists.append(float(emd_distance(p, q, ground=ground,
                                        hierarchy=hierarchy, domain=dom)))
        sizes.append(len(blocks[k]))

    worst = max(dists) if dists else 0.0
    return RichResult(payload={
        "estimate": float(worst),
        "achieved_t": float(worst),
        "satisfies": bool(worst <= t),
        "class_distances": dists,
        "class_sizes": sizes,
        "n_classes": len(keys),
        "min_class_size": int(min(sizes)) if sizes else 0,
        "overall_distribution": q,
        "domain": [str(v) for v in dom],
        "ground": str(ground).lower(),
        "t": t,
        "n": int(n),
        "method": "t-closeness via EMD (Li, Li & Venkatasubramanian 2007)",
    })


def cheatsheet():
    return ("tcls: t-closeness, EMD(class dist, table dist) <= t; grounds "
            "equal = half variational, ordered = sum|cumsum(p-q)|/(m-1), "
            "hierarchical = sum_N height(N)/H * min(pos,neg extra).")


tcls = t_closeness
