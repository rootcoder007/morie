# morie.fn -- function file (rootcoder007/morie)
r"""Deep InfoMax: maximise mutual information LOCALLY.

Maximising the mutual information between an input and its
representation is an old idea, and on its own it is a bad objective:
MI is invariant to invertible transformations, so a representation
that memorises pixel noise scores as well as one that captures
content.

**The fix is where the information is measured.** Deep InfoMax
maximises MI between the **global** summary vector and **local**
patches of the feature map -- so a representation is rewarded for
containing what is shared across many locations, and a feature
explaining a single patch of noise earns nothing. That is the paper's
central result: local structure in the input is what makes the
representation good for classification, and ``local_objective``
against ``global_objective`` measures the difference rather than
asserting it.

**The estimator matters too.** The Donsker-Varadhan bound on the KL
divergence has an expectation inside a logarithm, so its gradient is
biased and its variance explodes with the batch. The
**Jensen-Shannon** form,

.. math:: \hat I_{JSD} = E_{P}[-\mathrm{sp}(-T(x,y))]
          - E_{P\times\tilde P}[\mathrm{sp}(T(x',y))],

with :math:`\mathrm{sp}(z)=\log(1+e^z)`, is bounded, stable, and gives
better results -- the paper's own comparison, and the reason it is the
default here. Both are implemented, so the instability can be seen.

**One global feature, one estimator, one step.** Unlike CPC, which
processes local features sequentially and predicts the "future" of a
summary with separate estimators, DIM's single global feature predicts
all local features simultaneously.

References
----------
Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K.,
Bachman, P., Trischler, A. & Bengio, Y. (2019) "Learning deep
representations by mutual information estimation and maximization",
*International Conference on Learning Representations (ICLR 2019)*,
arXiv:1808.06670. Sec. 2-3: that maximising MI between the input and
output of an encoder can be done with an MI estimator; that a
Jensen-Shannon-divergence-based alternative to the Donsker-Varadhan
KL estimator is more stable and provides better results; that
structure-aware objectives -- maximising MI between the global feature
and LOCAL patches of the feature map -- improve the suitability of the
representation for classification; and the comparison with CPC, which
processes local features sequentially to build summary features and
predicts specific local features autoregressively with separate
estimators, whereas DIM uses a single global summary feature that
predicts all local features simultaneously in one step with one
estimator.

van den Oord, A., Li, Y. & Vinyals, O. (2018) "Representation
Learning with Contrastive Predictive Coding", arXiv:1807.03748. CPC,
the ordered-autoregression alternative.

Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y.,
Courville, A. & Hjelm, R. D. (2018) "Mutual Information Neural
Estimation", *ICML 2018*, PMLR 80, 531-540, arXiv:1801.04062. The
Donsker-Varadhan estimator being replaced.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["softplus", "jsd_estimator", "dv_estimator",
           "global_objective", "local_objective"]

_EPS = 1e-12


def softplus(z):
    r""":math:`\log(1+e^z)`, computed without overflowing."""
    v = float(z)
    return v + math.log1p(math.exp(-v)) if v > 0 else \
        math.log1p(math.exp(v))


def jsd_estimator(joint_scores, marginal_scores):
    r""":math:`E_P[-\mathrm{sp}(-T)] - E_{P\times\tilde P}
    [\mathrm{sp}(T)]`.

    Bounded, so its gradient does not explode with the batch -- the
    reason it is preferred to the KL form.
    """
    J = [float(v) for v in k.vec(joint_scores)]
    M = [float(v) for v in k.vec(marginal_scores)]
    if not J or not M:
        raise ValueError("infmax: both joint and marginal samples "
                         "are needed")
    pos = sum(-softplus(-v) for v in J) / len(J)
    neg = sum(softplus(v) for v in M) / len(M)
    return {"estimate": pos - neg, "positive": pos, "negative": neg,
            "bounded": True,
            "note": "each term is bounded by construction"}


def dv_estimator(joint_scores, marginal_scores):
    r"""Donsker-Varadhan: :math:`E_P[T] - \log E_{P\times\tilde P}
    [e^T]`.

    The expectation sits INSIDE a logarithm, so the gradient is biased
    and the variance grows with the score scale -- reported here so the
    instability is visible rather than argued.
    """
    J = [float(v) for v in k.vec(joint_scores)]
    M = [float(v) for v in k.vec(marginal_scores)]
    if not J or not M:
        raise ValueError("infmax: both joint and marginal samples "
                         "are needed")
    mx = max(M)
    lse = mx + math.log(sum(math.exp(v - mx) for v in M) / len(M))
    mean_m = sum(M) / len(M)
    var = sum((v - mean_m) ** 2 for v in M) / max(len(M) - 1, 1)
    return {"estimate": sum(J) / len(J) - lse,
            "log_sum_exp": lse, "negative_variance": var,
            "bounded": False,
            "note": "unbounded above; large scores dominate the "
                    "log-mean-exp"}


def global_objective(global_features, feature_maps, critic,
                     estimator="jsd"):
    r"""MI between the global vector and the WHOLE feature map.

    Invariant to any invertible transformation of the map, which is
    why it can be maximised by a representation that has learned
    nothing useful.
    """
    G = [[float(v) for v in r] for r in k.mat(global_features)]
    F = [[float(v) for v in k.vec(m)] for m in feature_maps]
    n = len(G)
    if len(F) != n:
        raise ValueError("infmax: %d globals but %d feature maps"
                         % (n, len(F)))
    if n < 2:
        raise ValueError("infmax: negatives come from other examples "
                         "in the batch, so at least 2 are needed")
    joint = [float(critic(G[i], F[i])) for i in range(n)]
    marg = [float(critic(G[i], F[j])) for i in range(n)
            for j in range(n) if i != j]
    est = (jsd_estimator if estimator == "jsd" else dv_estimator)
    r = est(joint, marg)
    return {"objective": r["estimate"], "estimator": estimator,
            "n_positive": len(joint), "n_negative": len(marg),
            "note": "one score per image; the spatial structure is "
                    "discarded"}


def local_objective(global_features, feature_maps, critic,
                    estimator="jsd"):
    r"""MI between the global vector and EACH LOCAL patch, averaged.

    A feature that explains one patch of noise scores nothing here,
    because it must pay off at every location -- which is the whole
    reason the local objective produces better classification
    features.
    """
    G = [[float(v) for v in r] for r in k.mat(global_features)]
    M = [[[float(v) for v in k.vec(p)] for p in m]
         for m in feature_maps]
    n = len(G)
    if len(M) != n:
        raise ValueError("infmax: %d globals but %d feature maps"
                         % (n, len(M)))
    if n < 2:
        raise ValueError("infmax: at least 2 examples are needed for "
                         "negatives")
    L = len(M[0])
    if any(len(m) != L for m in M):
        raise ValueError("infmax: the feature maps have differing "
                         "numbers of locations")
    joint, marg = [], []
    for i in range(n):
        for l in range(L):
            joint.append(float(critic(G[i], M[i][l])))
            for j in range(n):
                if j != i:
                    marg.append(float(critic(G[i], M[j][l])))
    est = (jsd_estimator if estimator == "jsd" else dv_estimator)
    r = est(joint, marg)
    return RichResult(payload={
        "estimate": r["estimate"], "objective": r["estimate"],
        "estimator": estimator, "n_locations": L,
        "n_positive": len(joint), "n_negative": len(marg),
        "method": "Deep InfoMax local objective; Hjelm et al. (2019)",
        "note": "the global feature predicts ALL locations at once, "
                "with ONE estimator and no autoregression",
    })


def cheatsheet():
    return ("infmax: maximising MI between input and representation is "
            "a bad objective alone -- MI is invariant to invertible "
            "maps, so memorising noise scores as well as capturing "
            "content. Measure it LOCALLY instead: between the global "
            "summary and each patch of the feature map, so a feature "
            "must pay off at many locations. Use the JENSEN-SHANNON "
            "estimator, -sp(-T) minus sp(T), which is BOUNDED, rather "
            "than Donsker-Varadhan, whose expectation sits inside a "
            "log and whose variance explodes. Unlike CPC there is ONE "
            "global feature, ONE estimator, and no autoregression.")


# compact alias per ledger/NAMING.md
deepinfomax = local_objective

# public names resolved by fn/_lazy_map.json
infomax_objective = local_objective
