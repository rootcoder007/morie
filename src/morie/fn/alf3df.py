"""AlphaFold-3 style diffusion sampling step on atom coordinates.

A structure-prediction diffusion model does not predict coordinates in
one shot. It starts from noise at a large scale and walks down a
schedule of decreasing noise levels, and at each level it asks a
denoiser "given this rattled structure, what was the clean one?" and
steps part of the way there. The step is what this module is: the
schedule, the churn, the update, and the coordinate augmentation that
makes the whole thing equivariant.

The update is the EDM sampler, written in the form where the denoiser
D(x, sigma) returns an estimate of the CLEAN coordinates rather than of
the noise. Then

    d = (x - D(x, sigma)) / sigma

is the direction of increasing noise, and stepping along it by
(sigma_next - sigma) walks down the schedule. That parameterisation has
a property worth stating because it is the module's strongest check: if
the denoiser is an oracle that returns the true structure, then a point
at noise level sigma lands at EXACTLY noise level sigma_next, and at the
end of the schedule, where sigma is zero, it lands exactly on the
answer. Not approximately -- exactly, in floating point.

Churn is the stochastic part. Rather than walking straight down, the
sampler can add noise back at each step and then denoise further, which
trades a little accuracy for diversity. With the churn parameter at zero
the sampler is fully deterministic and consumes no randomness at all,
which is worth being able to check.

The AlphaFold-3 addition is CENTRE RANDOM AUGMENTATION: before each
step the coordinates are recentred on their own centroid and given a
random rotation. Structure prediction should not care where in space the
molecule sits or how it is oriented, and augmenting during sampling is
how that invariance is enforced at inference rather than merely hoped
for. Centring makes the centroid exactly zero and rotation preserves
every interatomic distance exactly, and both are checked.

On constants. The AlphaFold-3 paper describes the sampler in the main
text but puts its hyperparameters in Supplementary Methods 3.7, which is
not in hand. So the schedule shape follows Karras et al., whose formula
is published in full and is quoted below, and every AlphaFold-3-specific
constant is a PARAMETER with no default pretending to be theirs. A
module that invented gamma0 and called it AlphaFold's would be worse
than one that asks.

References
  Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel,
    A., Ronneberger, O., Willmore, L., Ballard, A.J., Bambrick, J. et
    al. (2024) "Accurate structure prediction of biomolecular
    interactions with AlphaFold 3." Nature 630(8016), 493-500.
    doi:10.1038/s41586-024-07487-w. The diffusion module and the
    centre-random-augmentation; the sampler hyperparameters are in its
    Supplementary Methods 3.7 and are not reproduced here.
  Karras, T., Aittala, M., Aila, T. and Laine, S. (2022) "Elucidating
    the design space of diffusion-based generative models." Advances in
    Neural Information Processing Systems 35, 26565-26577.
    arXiv:2206.00364. The noise schedule (their equation 5), the
    denoiser parameterisation and the stochastic sampler (their
    Algorithm 2) implemented here.
  Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B. (2021) "Score-based generative modeling through
    stochastic differential equations." ICLR. The probability-flow view
    the update descends.
  Shoemake, K. (1992) "Uniform random rotations." In Graphics Gems III,
    124-132. The rotation used by the augmentation.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["alf3df", "af3_diffusion_step", "karras_schedule",
           "random_rotation", "centre_random_augmentation", "sample",
           "ORDERS", "cheatsheet"]

ORDERS = ("heun", "euler")


def karras_schedule(n_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """The EDM noise schedule, Karras et al. equation 5.

    sigma_i = (sigma_max^(1/rho)
               + i/(N-1) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    with a final zero appended. rho controls how much of the schedule is
    spent at low noise: larger puts more steps near the end, where the
    structure is actually being decided.
    """
    n = int(n_steps)
    if n < 2:
        raise ValueError("need at least two noise levels")
    if sigma_min <= 0.0 or sigma_max <= sigma_min:
        raise ValueError("need 0 < sigma_min < sigma_max")
    a = sigma_max ** (1.0 / rho)
    b = sigma_min ** (1.0 / rho)
    out = []
    for i in range(n):
        out.append((a + i * (b - a) / (n - 1.0)) ** rho)
    out.append(0.0)
    return out


def random_rotation(rng):
    """A uniform random rotation matrix by Shoemake's quaternion method.

    Three uniforms in, one rotation out, with no rejection step -- so it
    consumes exactly three draws whatever it returns and two
    implementations stay in step.
    """
    u1 = float(rng.uniform())
    u2 = float(rng.uniform())
    u3 = float(rng.uniform())
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    t1 = 2.0 * math.pi * u2
    t2 = 2.0 * math.pi * u3
    x = s1 * math.sin(t1)
    y = s1 * math.cos(t1)
    z = s2 * math.sin(t2)
    w = s2 * math.cos(t2)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ]


def centre_random_augmentation(x, rng):
    """Recentre on the centroid, then rotate at random.

    The centring is exact: the returned centroid is zero to rounding,
    not approximately zero. The rotation preserves every interatomic
    distance, which is the property that makes the augmentation
    harmless to the structure and useful to the model.
    """
    n = len(x)
    if n == 0:
        return [], [0.0, 0.0, 0.0]
    cen = [_w.csum(row[t] for row in x) / n for t in range(3)]
    R = random_rotation(rng)
    out = []
    for row in x:
        c = [row[t] - cen[t] for t in range(3)]
        out.append([_w.dot(R[t], c) for t in range(3)])
    return out, cen


def af3_diffusion_step(x, t, score_fn, sigma_next=None, gamma=0.0,
                       noise_scale=1.0, step_scale=1.0, order="heun",
                       rng=None, augment=False):
    """One step of the sampler, from noise level t down to sigma_next.

    Parameters
    ----------
    x : sequence of sequences
        Current coordinates, one row per atom, three columns.
    t : float
        The current noise level sigma.
    score_fn : callable
        The denoiser D(x, sigma), returning an estimate of the CLEAN
        coordinates. Note the parameterisation: not a score, not a
        noise prediction.
    sigma_next : float or None
        The level to step to. Zero when omitted, which is the final
        step.
    gamma : float
        Churn. The level is raised to t(1 + gamma) and the difference
        added as noise before denoising. Zero is deterministic.
    noise_scale, step_scale : float
        The AlphaFold-3 sampler scales the injected noise and the step
        itself. Their published values live in Supplementary Methods
        3.7 and are not reproduced here, so these default to one --
        which is the plain EDM sampler, not AlphaFold's.
    order : str
        "heun" applies the second-order correction, "euler" does not.

    Returns
    -------
    dict
        The stepped coordinates, the churned level, and the direction.

    References
    ----------
    Karras et al. (2022) NeurIPS 35, Algorithm 2; Abramson et al. (2024)
    Nature 630(8016), 493-500.
    """
    if order not in ORDERS:
        raise ValueError("order must be one of %r" % (ORDERS,))
    t = float(t)
    if t <= 0.0:
        raise ValueError("the current noise level must be positive")
    sn = 0.0 if sigma_next is None else float(sigma_next)
    if sn < 0.0 or sn > t:
        raise ValueError("the next level must lie in [0, t]")
    cur = [[float(v) for v in row] for row in x]
    if augment:
        if rng is None:
            raise ValueError("the augmentation needs a random stream")
        cur, _ = centre_random_augmentation(cur, rng)

    that = t * (1.0 + float(gamma))
    if gamma > 0.0:
        if rng is None:
            raise ValueError("churn needs a random stream")
        amt = math.sqrt(that * that - t * t) * float(noise_scale)
        cur = [[row[c] + amt * float(rng.normal()) for c in range(3)]
               for row in cur]

    den = score_fn(cur, that)
    d = [[(cur[i][c] - den[i][c]) / that for c in range(3)]
         for i in range(len(cur))]
    dt = (sn - that) * float(step_scale)
    nxt = [[cur[i][c] + dt * d[i][c] for c in range(3)]
           for i in range(len(cur))]
    if order == "heun" and sn > 0.0:
        # The second-order correction: average the direction at the
        # start and the end of the step. It costs one more denoiser call
        # and is what makes the sampler second order rather than first.
        den2 = score_fn(nxt, sn)
        d2 = [[(nxt[i][c] - den2[i][c]) / sn for c in range(3)]
              for i in range(len(nxt))]
        nxt = [[cur[i][c] + dt * 0.5 * (d[i][c] + d2[i][c])
                for c in range(3)] for i in range(len(cur))]
    return {"x": nxt, "sigma_hat": that, "direction": d}


def sample(shape_n, score_fn, n_steps=8, sigma_min=0.002, sigma_max=80.0,
           rho=7.0, gamma=0.0, noise_scale=1.0, step_scale=1.0,
           order="heun", seed=0, augment=False):
    """Run the whole schedule from noise to a structure."""
    sig = karras_schedule(n_steps, sigma_min, sigma_max, rho)
    rng = _core._SplitMix64(seed)
    n = int(shape_n)
    x = [[sig[0] * float(rng.normal()) for _ in range(3)]
         for _ in range(n)]
    traj = []
    for i in range(len(sig) - 1):
        r = af3_diffusion_step(x, sig[i], score_fn, sig[i + 1], gamma,
                               noise_scale, step_scale, order, rng,
                               augment)
        x = r["x"]
        traj.append(r["sigma_hat"])
    return x, sig, traj


def alf3df(x, t, score_fn, **kw):
    """The ledger entry point: one step, reported richly."""
    r = af3_diffusion_step(x, t, score_fn, **kw)
    nx = r["x"]
    n = len(nx)
    cen = [_w.csum(row[c] for row in nx) / n for c in range(3)]
    rad = math.sqrt(_w.csum(
        (nx[i][c] - cen[c]) * (nx[i][c] - cen[c])
        for i in range(n) for c in range(3)) / n) if n else float("nan")
    return RichResult(payload={
        "x": nx,
        "direction": r["direction"],
        "sigma_hat": r["sigma_hat"],
        "sigma": float(t),
        "centroid": cen,
        "radius_of_gyration": rad,
        "estimate": rad,
        "se": float("nan"),
        "n_atoms": n,
        "method": "AlphaFold-3 style diffusion sampling step",
    })


def cheatsheet():
    return ("alf3df: AlphaFold-3 style diffusion step. orders "
            + ", ".join(ORDERS)
            + "; Karras schedule and update, centre-random augmentation, "
              "AF3 constants supplied by the caller")
