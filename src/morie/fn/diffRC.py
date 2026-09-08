# morie.fn -- function file (rootcoder007/morie)
r"""DiffRec: diffusion over interactions, with the noise turned down.

Generative recommenders had come in two shapes, and both distort. A
GAN-based recommender optimises a minimax objective and is unstable; a
VAE-based one trades tractability for a bound and, being a *tractable*
approximation, tends to blur the very signal it is meant to model.
Diffusion offers a third shape -- but the schedule that works for
images is wrong here.

**Why the standard schedule is wrong.** Image diffusion corrupts
:math:`x_0` until it is pure noise, because the sampler must start
from noise. A user's interaction history is not something to destroy:
corrupting it completely erases the personalised signal that *is* the
prediction target. DiffRec therefore adds noise on a **reduced**
scale, keeping the personal history recoverable, and
``noise_schedule`` exposes that scale as the knob it is.

**The forward process is closed form and must be checked as one.**

.. math:: q(x_t \mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}x_0,\;
          (1-\bar\alpha_t)I\big),\qquad
          \bar\alpha_t = \prod_{s\le t}(1-\beta_s),

so the mean and variance at any :math:`t` follow from
:math:`\bar\alpha_t` alone -- no simulation needed, which is what the
anchor exploits.

**At noise scale zero the model must be the identity.** Not
approximately: :math:`\bar\alpha_t = 1` for every :math:`t`, the
forward process does nothing, and the reverse returns the history
unchanged. That is the boundary condition a schedule bug breaks first.

**Importance sampling over :math:`t`** concentrates training effort on
the timesteps with the largest loss instead of drawing uniformly, so
the steps that matter are not visited by luck.

References
----------
Wang, W., Xu, Y., Feng, F., Lin, X., He, X. & Chua, T.-S. (2023)
"Diffusion Recommender Model", *Proceedings of the 46th International
ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR '23)*, 832-841, doi:10.1145/3539618.3591663,
arXiv:2304.04971. The criticism of GAN-based recommenders (unstable
adversarial training) and VAE-based ones (a trade-off between
tractability and representation ability); the diffusion formulation
over user interaction histories; the reduced noise scale in the
forward process to avoid corrupting the personalised information in a
user's interactions; and importance sampling over diffusion steps for
training.

Ho, J., Jain, A. & Abbeel, P. (2020) "Denoising Diffusion
Probabilistic Models", *NeurIPS 2020*, arXiv:2006.11239. The forward
process and the closed form used above.

Liang, D., Krishnan, R. G., Hoffman, M. D. & Jebara, T. (2018)
"Variational Autoencoders for Collaborative Filtering", *WWW 2018*,
689-698, arXiv:1802.05814. The VAE recommender being displaced.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["noise_schedule", "forward_corrupt", "posterior_mean",
           "importance_weights", "denoise"]

_EPS = 1e-12


def noise_schedule(T, scale=0.001, beta_min=0.0001, beta_max=0.02):
    r"""A linear :math:`\beta` schedule, scaled DOWN.

    ``scale`` is the whole point: at image-diffusion scales the
    history is destroyed, and with it the thing being predicted.
    """
    n = int(T)
    s = float(scale)
    if n < 1:
        raise ValueError("diffRC: T must be at least 1")
    if s < 0.0:
        raise ValueError("diffRC: the noise scale cannot be "
                         "negative")
    betas, abar, acc = [], [], 1.0
    for t in range(n):
        b = s * (float(beta_min) + (float(beta_max) - float(beta_min))
                 * (t / float(max(n - 1, 1))))
        betas.append(b)
        acc *= (1.0 - b)
        abar.append(acc)
    return {"beta": betas, "alpha_bar": abar, "T": n, "scale": s,
            "signal_retained": abar[-1],
            "note": "scale = 0 leaves alpha_bar at 1, so the forward "
                    "process is the identity"}


def forward_corrupt(x0, alpha_bar_t, rng=None):
    r""":math:`q(x_t\mid x_0)`: mean :math:`\sqrt{\bar\alpha_t}x_0`,
    variance :math:`1-\bar\alpha_t`.

    Closed form, so no simulation is needed to check it.
    """
    x = [float(v) for v in k.vec(x0)]
    ab = float(alpha_bar_t)
    if not 0.0 <= ab <= 1.0:
        raise ValueError("diffRC: alpha_bar must lie in [0,1], got "
                         "%r" % (alpha_bar_t,))
    sm = math.sqrt(ab)
    sv = math.sqrt(max(1.0 - ab, 0.0))
    mean = [sm * v for v in x]
    if rng is None or sv <= _EPS:
        return {"x_t": mean, "mean": mean, "std": sv,
                "sampled": False}
    xt = [mean[i] + sv * (2.0 * float(rng.uniform()) - 1.0)
          * math.sqrt(3.0) for i in range(len(x))]
    return {"x_t": xt, "mean": mean, "std": sv, "sampled": True}


def posterior_mean(x_t, x0_hat, alpha_bar_t, alpha_bar_prev, beta_t):
    r"""The DDPM posterior mean :math:`\mu(x_t, \hat x_0)`."""
    xt = [float(v) for v in k.vec(x_t)]
    x0 = [float(v) for v in k.vec(x0_hat)]
    if len(xt) != len(x0):
        raise ValueError("diffRC: x_t and the estimate of x_0 differ "
                         "in length")
    ab, abp, b = (float(alpha_bar_t), float(alpha_bar_prev),
                  float(beta_t))
    denom = 1.0 - ab
    if denom <= _EPS:
        return {"mean": list(x0), "degenerate": True,
                "note": "alpha_bar = 1: nothing was added, so nothing "
                        "is removed"}
    c0 = math.sqrt(max(abp, 0.0)) * b / denom
    ct = math.sqrt(max(1.0 - b, 0.0)) * (1.0 - abp) / denom
    return {"mean": [c0 * x0[i] + ct * xt[i] for i in range(len(xt))],
            "coef_x0": c0, "coef_xt": ct, "degenerate": False}


def importance_weights(step_losses, uniform=False, smoothing=0.1):
    r"""Sample the timesteps that actually carry loss.

    Uniform sampling spends most of its budget where the model is
    already right.
    """
    L = [float(v) for v in k.vec(step_losses)]
    if not L:
        raise ValueError("diffRC: no per-step losses given")
    if any(v < 0.0 for v in L):
        raise ValueError("diffRC: a loss cannot be negative")
    n = len(L)
    if uniform:
        return {"weights": [1.0 / n] * n, "uniform": True,
                "effective_steps": float(n)}
    s = float(smoothing)
    w = [math.sqrt(v) + s for v in L]
    z = sum(w)
    p = [v / z for v in w]
    eff = 1.0 / sum(v * v for v in p)
    return {"weights": p, "uniform": False,
            "effective_steps": eff,
            "note": "effective sample size falls as the loss "
                    "concentrates, which is the intended behaviour"}


def denoise(x_t, model, schedule, t_start=None):
    r"""Run the reverse chain from :math:`t` back to 0.

    With ``scale = 0`` the schedule is the identity and this returns
    its input unchanged -- the boundary condition a schedule bug
    breaks first.
    """
    x = [float(v) for v in k.vec(x_t)]
    ab = schedule["alpha_bar"]
    beta = schedule["beta"]
    T = schedule["T"]
    t = T - 1 if t_start is None else int(t_start)
    if t < 0 or t >= T:
        raise ValueError("diffRC: t is outside the schedule")
    path = []
    while t >= 0:
        x0h = [float(v) for v in k.vec(model(x, t))]
        prev = ab[t - 1] if t > 0 else 1.0
        pm = posterior_mean(x, x0h, ab[t], prev, beta[t])
        x = pm["mean"]
        path.append(list(x))
        t -= 1
    return RichResult(payload={
        "estimate": x, "x0": x, "path": path, "steps": len(path),
        "signal_retained": schedule["signal_retained"],
        "method": "diffusion over interaction histories; Wang et al. "
                  "(2023)",
        "note": "the noise scale is REDUCED so the personalised "
                "history survives the forward process",
    })


def cheatsheet():
    return ("diffRC: GAN recommenders are unstable and VAE ones trade "
            "representation for tractability, so use diffusion -- but "
            "NOT the image schedule. Image diffusion destroys x_0 "
            "because sampling starts from noise; a user's interaction "
            "history is the personalised signal being predicted, so "
            "corrupting it fully erases the target. DiffRec adds noise "
            "on a REDUCED scale. Forward is closed form: mean "
            "sqrt(alpha_bar) x_0, variance 1 - alpha_bar. At scale 0 "
            "the whole thing must be the IDENTITY. Timesteps are drawn "
            "by IMPORTANCE SAMPLING, so the steps that carry loss are "
            "not visited by luck.")


# compact alias per ledger/NAMING.md
diffusionrecommender = denoise

# public names resolved by fn/_lazy_map.json
diffusion_rec = denoise
diffusionrec = denoise
