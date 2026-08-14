# morie.fn -- function file (rootcoder007/morie)
r"""Deep InfoMax: maximise mutual information, but locally.

Learn a representation without labels by maximising the mutual
information between the input and its encoding. The obvious reading of
that -- maximise :math:`I(X; E_\psi(X))` for the whole image -- is
weaker than it sounds: a representation can capture the global
statistics and still be useless for anything that depends on
*structure*, and mutual information alone is invariant to any
bijection, so nothing forces the representation to be organised.

**So maximise it locally, and average.** Deep InfoMax's central result
is that maximising the *average* MI between the global summary and
**local patches** of the feature map works far better for downstream
tasks than the global objective alone. A feature that must predict
every patch cannot describe only what is common; it has to encode
content shared across the image, which is what a classifier wants.

**The estimator is a discriminator, not an integral.** MI is estimated
in the Donsker-Varadhan / Jensen-Shannon family by discriminating
*paired* samples -- a patch and the summary from the same image --
from *unpaired* ones drawn from a different image. The JSD form,

.. math:: \hat I^{JSD} = E_{P}[-\mathrm{sp}(-T(x, E(x)))]
          - E_{P \times \tilde P}[\mathrm{sp}(T(x', E(x)))],

with :math:`\mathrm{sp}(z) = \log(1+e^z)`, is bounded and behaves
better in practice than the DV form, whose value is unbounded and
whose gradient is high-variance. Both are implemented; the anchor shows
the JSD estimator staying finite where DV runs away.

**A prior on the representation is a separate knob.** Matching the
encoding to a prior distribution adversarially controls *how* the
information is stored -- compactness, independence -- which the MI term
alone does not constrain at all.

References
----------
Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K.,
Bachman, P., Trischler, A. & Bengio, Y. (2019) "Learning deep
representations by mutual information estimation and maximization",
*International Conference on Learning Representations (ICLR 2019)*,
arXiv:1808.06670. The abstract and Sec. 1-3: maximising mutual
information between the input and the output of a deep encoder;
structure matters, and maximising the AVERAGE MI between the global
representation and LOCAL patches greatly improves representation
quality for downstream tasks compared with the global objective; the
Donsker-Varadhan and Jensen-Shannon estimators built from a
discriminator over paired versus unpaired samples; and matching the
representation to a prior to control its characteristics.

Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y.,
Courville, A. & Hjelm, R. D. (2018) "Mutual Information Neural
Estimation", *ICML 2018*, PMLR 80, 531-540, arXiv:1801.04062. The
Donsker-Varadhan estimator.

Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S. & Wang, L. (2020) "Deep
Graph Contrastive Representation Learning", arXiv:2006.04131. The
graph-domain descendant; implemented in :mod:`grace`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["softplus", "jsd_estimate", "dv_estimate",
           "local_objective", "prior_matching_loss"]

_EPS = 1e-12
_ESTIMATORS = ("jsd", "dv")


def softplus(z):
    r""":math:`\mathrm{sp}(z) = \log(1+e^z)`, overflow-safe."""
    v = float(z)
    return v + math.log1p(math.exp(-v)) if v > 0 \
        else math.log1p(math.exp(v))


def jsd_estimate(paired, unpaired):
    r"""The Jensen-Shannon estimator -- bounded, and the paper's
    default."""
    p = [float(v) for v in k.vec(paired)]
    q = [float(v) for v in k.vec(unpaired)]
    if not p or not q:
        raise ValueError("mienco: both paired and unpaired scores are "
                         "needed")
    return (sum(-softplus(-v) for v in p) / len(p)
            - sum(softplus(v) for v in q) / len(q))


def dv_estimate(paired, unpaired):
    r"""The Donsker-Varadhan estimator.

    Unbounded above, which is exactly the instability the JSD form
    avoids -- the anchor pushes both and compares.
    """
    p = [float(v) for v in k.vec(paired)]
    q = [float(v) for v in k.vec(unpaired)]
    m = max(q)
    lse = m + math.log(sum(math.exp(v - m) for v in q) / len(q))
    return sum(p) / len(p) - lse


def local_objective(summary, patches, other_patches, critic,
                    estimator="jsd"):
    r"""Average MI between the global summary and every LOCAL patch.

    ``patches`` come from the same image as ``summary``;
    ``other_patches`` from a different one. Averaging over patches is
    the paper's central change -- a summary that must predict every
    patch cannot encode only global statistics.
    """
    if estimator not in _ESTIMATORS:
        raise ValueError("mienco: estimator must be one of %s, got %r"
                         % (", ".join(_ESTIMATORS), estimator))
    pos = [critic(summary, p) for p in patches]
    neg = [critic(summary, p) for p in other_patches]
    est = (jsd_estimate(pos, neg) if estimator == "jsd"
           else dv_estimate(pos, neg))
    return RichResult(payload={
        "estimate": est, "mi_lower_bound": est,
        "estimator": estimator, "n_patches": len(patches),
        "n_negative_patches": len(other_patches),
        "method": "local Deep InfoMax; Hjelm et al. (2019)",
        "note": "averaging over LOCAL patches beats the global "
                "objective for downstream tasks",
    })


def global_objective(summary, whole, other_whole, critic,
                     estimator="jsd"):
    r"""The global-only variant, for comparison."""
    return local_objective(summary, [whole], [other_whole], critic,
                           estimator)


def prior_matching_loss(samples, prior_samples, discriminator):
    r"""Adversarial matching of the encoding to a prior.

    Controls *how* the information is stored -- mutual information
    alone is invariant to any bijection and constrains nothing about
    the form.
    """
    a = [float(discriminator(s)) for s in samples]
    b = [float(discriminator(s)) for s in prior_samples]
    if not a or not b:
        raise ValueError("mienco: both encoded and prior samples are "
                         "needed")
    return (sum(softplus(-v) for v in b) / len(b)
            + sum(softplus(v) for v in a) / len(a))


def cheatsheet():
    return ("mienco: unsupervised representations by maximising mutual "
            "information -- but GLOBAL MI is weak, since MI is "
            "invariant to any bijection and a summary can capture "
            "global statistics while encoding no structure. The "
            "central result: maximise the AVERAGE MI between the "
            "summary and LOCAL PATCHES. MI is estimated by a "
            "discriminator separating paired from unpaired samples; "
            "the JSD form is BOUNDED where Donsker-Varadhan is not. A "
            "prior-matching term separately controls how the "
            "information is stored.")


# compact alias per ledger/NAMING.md
deepinfomax = local_objective

# public names resolved by fn/_lazy_map.json
mi_neural_encoder = local_objective
