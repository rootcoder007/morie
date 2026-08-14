# morie.fn -- function file (rootcoder007/morie)
r"""AnoGAN: anomaly detection by inverting a generator.

Anomalies are, by construction, what the training data does not
contain. So train a GAN on **normal** data only, and it learns a
manifold of normal appearance. A new image is then scored by how well
it can be *reproduced* from that manifold.

**The inversion is the method.** A GAN maps :math:`z \to G(z)` but
provides no inverse, so the latent code for a query image is found by
optimisation: fix the trained generator and descend on :math:`z` to
minimise a loss against the query. Nothing about the network changes;
the search is over the latent space alone, which is why an anomalous
image cannot simply be memorised.

**Two loss terms doing different work.**

* **Residual loss** :math:`\sum |x - G(z)|` -- pixel disagreement.
* **Discrimination loss** :math:`\sum |f(x) - f(G(z))|`, comparing
  *intermediate discriminator features* rather than its output.
  Comparing the discriminator's scalar verdict instead would give
  almost no gradient; the feature layer is what makes the term
  informative.

The score is :math:`(1-\lambda)L_R + \lambda L_D`, and the **residual
map** :math:`|x - G(z)|` localises the anomaly rather than only
flagging the image -- which is the clinically useful part.

**The failure mode is a generator that is too good.** If :math:`G`
can reproduce anything, every image scores zero and nothing is
anomalous. Restricted capacity is therefore load-bearing, not an
implementation compromise, and ``score_separation`` measures whether
normal and anomalous scores actually separate rather than assuming it.

References
----------
Schlegl, T., Seebock, P., Waldstein, S. M., Schmidt-Erfurth, U. &
Langs, G. (2017) "Unsupervised Anomaly Detection with Generative
Adversarial Networks to Guide Marker Discovery", *Information
Processing in Medical Imaging (IPMI 2017)*, LNCS 10265, 146-157,
doi:10.1007/978-3-319-59050-9_12, arXiv:1703.05921. Training a GAN on
normal data to learn a manifold of normal anatomical variability;
mapping a query image back to the latent space by iterative
optimisation of z with the generator fixed; the anomaly score
combining a residual loss on pixel differences with a discrimination
loss on intermediate discriminator features; and the residual image
localising anomalies.

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B.,
Warde-Farley, D., Ozair, S., Courville, A. & Bengio, Y. (2014)
"Generative Adversarial Nets", *NIPS 2014*, 2672-2680,
arXiv:1406.2661.

Radford, A., Metz, L. & Chintala, S. (2016) "Unsupervised
Representation Learning with Deep Convolutional Generative Adversarial
Networks", *ICLR 2016*, arXiv:1511.06434. The DCGAN architecture used.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["residual_loss", "discrimination_loss", "anomaly_score",
           "invert_to_latent", "residual_map", "score_separation"]

_EPS = 1e-12


def residual_loss(x, g_z):
    r""":math:`\sum |x - G(z)|` -- pixel disagreement."""
    a = [float(v) for v in k.vec(x)]
    b = [float(v) for v in k.vec(g_z)]
    if len(a) != len(b):
        raise ValueError("gan_an: the query and reconstruction "
                         "differ in size (%d, %d)" % (len(a), len(b)))
    return sum(abs(a[i] - b[i]) for i in range(len(a)))


def discrimination_loss(f_x, f_gz):
    r""":math:`\sum |f(x) - f(G(z))|` on INTERMEDIATE features.

    Not the discriminator's output: a scalar verdict gives almost no
    gradient, while the feature layer describes *how* the two differ.
    """
    a = [float(v) for v in k.vec(f_x)]
    b = [float(v) for v in k.vec(f_gz)]
    if len(a) != len(b):
        raise ValueError("gan_an: the feature vectors differ in size")
    if len(a) < 2:
        raise ValueError("gan_an: a scalar discriminator output "
                         "carries no gradient; use an intermediate "
                         "feature layer")
    return sum(abs(a[i] - b[i]) for i in range(len(a)))


def anomaly_score(x, g_z, f_x, f_gz, lam=0.1):
    r""":math:`(1-\lambda)L_R + \lambda L_D`."""
    l = float(lam)
    if not 0.0 <= l <= 1.0:
        raise ValueError("gan_an: lambda must lie in [0,1], got %r"
                         % (lam,))
    lr = residual_loss(x, g_z)
    ld = discrimination_loss(f_x, f_gz)
    return {"score": (1.0 - l) * lr + l * ld,
            "residual": lr, "discrimination": ld, "lambda": l}


def invert_to_latent(x, generator, feature_fn, z_dim, steps=200,
                     lr=0.05, lam=0.1, seed=0, h=1e-4,
                     step_decay=0.05):
    r"""Search the latent space with the generator FIXED.

    Only :math:`z` moves. Because the network cannot adapt, an image
    off the learned manifold stays badly reconstructed however long
    the search runs.

    Both loss terms are :math:`L_1`, so the subgradient does not shrink
    as the optimum is approached: a FIXED step oscillates around it
    forever at an amplitude set by the step size, and the last iterate
    is not the best one. Hence a decaying step
    :math:`\eta_t = \eta/(1 + t\,\text{step\_decay})` -- the
    standard subgradient schedule -- and the BEST iterate is returned
    rather than the last. ``step_decay=0`` restores the fixed step.
    """
    rng = np.random.default_rng(seed)
    z = [(float(rng.uniform()) - 0.5) * 2.0 for _ in range(int(z_dim))]
    fx = [float(v) for v in k.vec(feature_fn(x))]
    hist = []

    def loss(zv):
        g = generator(zv)
        return anomaly_score(x, g, fx, feature_fn(g), lam)["score"]

    best_z, best = list(z), loss(z)
    for t in range(int(steps)):
        base = loss(z)
        hist.append(base)
        if base < best:
            best, best_z = base, list(z)
        grad = []
        for i in range(len(z)):
            up = list(z)
            up[i] += h
            grad.append((loss(up) - base) / h)
        eta = float(lr) / (1.0 + t * float(step_decay))
        z = [z[i] - eta * grad[i] for i in range(len(z))]
    if loss(z) < best:
        best, best_z = loss(z), list(z)
    z = best_z
    g = generator(z)
    fin = anomaly_score(x, g, fx, feature_fn(g), lam)
    return RichResult(payload={
        "estimate": fin["score"], "score": fin["score"], "z": z,
        "reconstruction": g, "loss_history": hist,
        "residual": fin["residual"],
        "discrimination": fin["discrimination"],
        "final_step": float(lr) / (1.0 + (steps - 1) * float(step_decay)),
        "method": "AnoGAN latent inversion; Schlegl et al. (2017)",
        "note": "the generator is FIXED; only z moves, so an "
                "off-manifold image cannot be memorised",
    })


def residual_map(x, g_z, shape=None):
    r""":math:`|x - G(z)|` per pixel -- WHERE the anomaly is.

    Localisation, not just detection, which is the clinically useful
    output.
    """
    a = [float(v) for v in k.vec(x)]
    b = [float(v) for v in k.vec(g_z)]
    if len(a) != len(b):
        raise ValueError("gan_an: the query and reconstruction "
                         "differ in size")
    r = [abs(a[i] - b[i]) for i in range(len(a))]
    if shape is not None:
        h, w = int(shape[0]), int(shape[1])
        if h * w != len(r):
            raise ValueError("gan_an: the shape %dx%d does not match "
                             "%d values" % (h, w, len(r)))
        r = [r[i * w:(i + 1) * w] for i in range(h)]
    return {"map": r, "max": max(k.vec(r) if shape is None
                                 else [v for row in r for v in row]),
            "note": "localises the anomaly rather than only flagging "
                    "the image"}


def score_separation(normal_scores, anomalous_scores):
    r"""Do the two populations actually separate?

    A generator good enough to reproduce anything scores everything
    zero -- restricted capacity is load-bearing, so this must be
    measured.
    """
    a = [float(v) for v in k.vec(normal_scores)]
    b = [float(v) for v in k.vec(anomalous_scores)]
    if not a or not b:
        raise ValueError("gan_an: both populations are needed")
    hits = sum(1 for x in a for y in b if y > x)
    auc = hits / float(len(a) * len(b))
    return {"auc": auc, "mean_normal": sum(a) / len(a),
            "mean_anomalous": sum(b) / len(b),
            "separated": auc > 0.7,
            "note": "an over-capable generator reconstructs anomalies "
                    "too and collapses this to 0.5"}


def cheatsheet():
    return ("gan_an: train a GAN on NORMAL data only, then score a "
            "query by how well it can be reproduced from that "
            "manifold. A GAN has no inverse, so find z by OPTIMISATION "
            "with the generator FIXED -- nothing adapts, so an "
            "off-manifold image stays badly reconstructed. Two losses: "
            "pixel residual, and a discrimination loss on INTERMEDIATE "
            "discriminator features (the scalar verdict would give no "
            "gradient). The residual map LOCALISES the anomaly. A "
            "generator that can reproduce anything scores everything "
            "zero -- limited capacity is load-bearing.")


# compact alias per ledger/NAMING.md
anogan = invert_to_latent

# public names resolved by fn/_lazy_map.json
gan_anomaly = invert_to_latent
gananomaly = invert_to_latent
