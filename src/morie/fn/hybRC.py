# morie.fn -- function file (rootcoder007/morie)
r"""Hybrid recommenders: seven ways to combine, and they differ.

Collaborative filtering cannot recommend an item nobody has rated --
the **new-item** or cold-start problem -- and content-based filtering
cannot surprise anyone, since it only returns more of what the user
already liked. Each is strong where the other is weak, so combining
them is obvious; *how* to combine them is not, and Burke's point is
that the choices are genuinely different systems, not variations on
one.

**The seven methods**, each implemented here:

* **Weighted** -- scores from several recommenders are combined
  numerically. Simple, and it assumes the relative value of the
  components is roughly uniform across the item space, which is
  exactly what fails when one component is blind to new items.
* **Switching** -- pick one recommender per case by some criterion.
  Buys sensitivity to each component's strengths at the cost of a new
  layer of parameterisation: the criterion itself.
* **Mixed** -- present results from several side by side, no fusion.
* **Feature combination** -- treat collaborative data as extra
  *features* inside a single content-based algorithm.
* **Cascade** -- one recommender ranks, the next breaks ties only.
  Strictly ordered, so it is not commutative.
* **Feature augmentation** -- one produces a feature that becomes
  input to the next.
* **Meta-level** -- one produces a whole *model* that is the next
  one's input, which is a stronger coupling than a feature.

**Order matters for some and not others**, and the module says which:
weighted, mixed, switching and feature-combination are
order-insensitive, so a CN/CF system is the same as CF/CN; cascade,
feature augmentation and meta-level are not. ``is_order_sensitive``
makes that checkable, and the anchor verifies cascade actually changes
under a swap while weighted does not.

References
----------
Burke, R. (2002) "Hybrid Recommender Systems: Survey and
Experiments", *User Modeling and User-Adapted Interaction* 12(4),
331-370, doi:10.1023/A:1021240730564. [PDF supplied by Vee.] The
taxonomy of seven hybridisation methods -- weighted, switching, mixed,
feature combination, cascade, feature augmentation and meta-level --
with the weighted hybrid's implicit assumption that the relative value
of the techniques is more or less uniform across the space of possible
items; the note that switching hybrids introduce additional complexity
because the switching criteria must be determined, adding another
level of parameterisation, in exchange for sensitivity to the
components' strengths and weaknesses; and the observation that four
techniques -- weighted, mixed, switching and feature combination --
are order-insensitive, so a CN/CF mixed system is no different from a
CF/CN one.

Balabanovic, M. & Shoham, Y. (1997) "Fab: content-based,
collaborative recommendation", *Communications of the ACM* 40(3),
66-72, doi:10.1145/245108.245124. An early hybrid.

Resnick, P. et al. (1994) "GroupLens", *CSCW '94*, 175-186,
doi:10.1145/192844.192905. The collaborative half; implemented in
:mod:`ucfR`.
"""

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["weighted", "switching", "mixed", "feature_combination",
           "cascade", "feature_augmentation", "meta_level",
           "is_order_sensitive", "METHODS"]

_EPS = 1e-12

METHODS = ("weighted", "switching", "mixed", "feature_combination",
           "cascade", "feature_augmentation", "meta_level")
_ORDER_INSENSITIVE = ("weighted", "switching", "mixed",
                      "feature_combination")


def is_order_sensitive(method):
    r"""Does swapping the two components give a different system?"""
    m = str(method)
    if m not in METHODS:
        raise ValueError("hybRC: method must be one of %s, got %r"
                         % (", ".join(METHODS), method))
    return {"method": m, "order_sensitive": m not in
            _ORDER_INSENSITIVE,
            "note": "weighted, mixed, switching and feature "
                    "combination are order-INsensitive; the other "
                    "three are pipelines"}


def weighted(scores, weights=None):
    r"""Combine scores numerically.

    Assumes each component is comparably good across the whole item
    space -- the assumption that breaks when one is blind to new
    items.
    """
    S = [dict(s) for s in scores]
    if not S:
        raise ValueError("hybRC: no component scores given")
    w = [1.0 / len(S)] * len(S) if weights is None else \
        [float(v) for v in k.vec(weights)]
    if len(w) != len(S):
        raise ValueError("hybRC: %d weight(s) for %d components"
                         % (len(w), len(S)))
    items = sorted(set().union(*[set(s) for s in S]))
    out, partial = {}, {}
    for it in items:
        present = [c for c in range(len(S)) if it in S[c]]
        out[it] = sum(w[c] * float(S[c][it]) for c in present)
        partial[it] = len(present) < len(S)
    return {"scores": out, "ranking": sorted(items,
                                             key=lambda i: -out[i]),
            "partially_scored": [i for i in items if partial[i]],
            "note": "an item missing from a component is scored by "
                    "the rest, which silently favours whoever HAS it"}


def switching(scores, criterion, context=None):
    r"""Pick ONE recommender per case.

    ``criterion`` returns the index to use. The criterion is itself a
    new set of parameters -- the cost the paper names.
    """
    S = [dict(s) for s in scores]
    c = int(criterion(context))
    if c < 0 or c >= len(S):
        raise ValueError("hybRC: the switching criterion chose "
                         "component %d of %d" % (c, len(S)))
    return {"scores": S[c], "chosen": c,
            "ranking": sorted(S[c], key=lambda i: -S[c][i]),
            "note": "sensitive to each component's strengths, at the "
                    "cost of another level of parameterisation"}


def mixed(recommendations, top_k=None):
    r"""Present several lists side by side; no fusion at all."""
    L = [list(r) for r in recommendations]
    if not L:
        raise ValueError("hybRC: no recommendation lists given")
    out = []
    for t in range(max(len(x) for x in L)):
        for src in range(len(L)):
            if t < len(L[src]):
                out.append({"item": L[src][t], "source": src})
    if top_k is not None:
        out = out[:int(top_k)]
    return {"presented": out, "n_sources": len(L),
            "note": "no score is combined, so no comparability "
                    "between components is assumed"}


def feature_combination(content_features, collaborative_features):
    r"""Collaborative data as EXTRA FEATURES in one content model."""
    C = [[float(v) for v in r] for r in k.mat(content_features)]
    D = [[float(v) for v in r] for r in k.mat(collaborative_features)]
    if len(C) != len(D):
        raise ValueError("hybRC: %d content rows but %d "
                         "collaborative rows" % (len(C), len(D)))
    return {"features": [C[i] + D[i] for i in range(len(C))],
            "content_dim": len(C[0]), "collaborative_dim": len(D[0]),
            "note": "one algorithm, wider input -- not two systems"}


def cascade(primary, secondary, tol=1e-9):
    r"""The second recommender breaks TIES only.

    Strictly ordered: the secondary can never overturn a strict
    preference of the primary.
    """
    P = dict(primary)
    S = dict(secondary)
    items = sorted(P)
    order = sorted(items, key=lambda i: (-P[i],
                                         -S.get(i, 0.0)))
    ties = {}
    for i in items:
        ties.setdefault(round(P[i] / max(tol, 1e-12)), []).append(i)
    broken = sum(1 for g in ties.values() if len(g) > 1)
    return {"ranking": order, "tie_groups_broken": broken,
            "primary_respected": all(
                P[order[a]] >= P[order[a + 1]] - tol
                for a in range(len(order) - 1)),
            "note": "the secondary NEVER overturns a strict "
                    "preference of the primary"}


def feature_augmentation(base_output, consumer):
    r"""One recommender's OUTPUT becomes the next one's input
    feature."""
    return {"result": consumer(base_output),
            "note": "a feature, not a model -- the consumer keeps its "
                    "own learning algorithm"}


def meta_level(model_builder, consumer, data):
    r"""One recommender's whole MODEL is the next one's input.

    Stronger coupling than augmentation: the consumer depends on the
    producer's internal representation, not just its output.
    """
    model = model_builder(data)
    return RichResult(payload={
        "estimate": consumer(model), "result": consumer(model),
        "model": model,
        "method": "meta-level hybrid; Burke (2002)",
        "note": "the consumer depends on the producer's internal "
                "representation, so the two cannot be swapped",
    })


def cheatsheet():
    return ("hybRC: collaborative filtering cannot recommend what "
            "nobody rated; content-based filtering cannot surprise "
            "anyone. Combining is obvious, HOW is not -- and the seven "
            "ways are different systems. WEIGHTED (assumes components "
            "are comparably good everywhere, which is what fails on "
            "new items), SWITCHING (buys sensitivity, costs a new "
            "criterion to parameterise), MIXED (side by side, no "
            "fusion), FEATURE COMBINATION (collaborative data as extra "
            "features in ONE model), CASCADE (second breaks TIES "
            "only), FEATURE AUGMENTATION (output feeds the next), "
            "META-LEVEL (whole MODEL feeds the next). The first four "
            "are order-insensitive; the last three are pipelines.")


# compact alias per ledger/NAMING.md
hybrid_recommender = weighted

# public names resolved by fn/_lazy_map.json
hybrid_rec = weighted
hybridrec = weighted
