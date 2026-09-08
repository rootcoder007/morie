# morie.fn -- function file (rootcoder007/morie)
r"""AlphaFold-3 co-folding: the diffusion sampler, and the confidence.

AlphaFold-3 predicts protein and ligand atoms together by denoising: start
from pure noise at a huge scale and walk down a schedule, letting a learned
module clean up the coordinates at each level. The learned module is the
network. The WALK is Algorithm 18 of the supplement and is exact
arithmetic, which is what this implements.

The schedule is the Karras one,

.. math:: \sigma_i = \sigma_{data}\left(s_{max}^{1/\rho}
          + \tfrac{i}{T}\left(s_{min}^{1/\rho}-s_{max}^{1/\rho}\right)
          \right)^{\rho},

and each step re-noises before it denoises: the level is inflated to
:math:`\hat t = \sigma_{i-1}(1+\gamma)` when :math:`\sigma_i` is above
``gamma_min``, noise of exactly the variance that inflation implies is
added, the module is asked for a clean structure, and the coordinates move
along that direction scaled by ``step_scale``. The churn is deliberate --
adding noise back before removing it is what lets the sampler escape a bad
early commitment.

Three ways in, as with the confidence heads:

**Sample** with a supplied ``denoiser`` -- any callable taking noisy
coordinates and a noise level and returning cleaned ones. That is the
route with a trained network.

**Fit** one: pass ``clean`` structures and a linear denoiser is trained
here by denoising score matching, ridge-solved in closed form at each
noise level rather than by gradient descent, because the objective is a
least squares problem and pretending otherwise wastes accuracy.

**Auto-tune** the schedule: ``sigma_data="fit"`` takes it from the spread
of the training coordinates instead of the paper's 16, which is the right
move when the coordinates are not in angstroms.

Randomness is the package's deterministic low-discrepancy normal stream,
not an RNG, so a run reproduces exactly and both language arms agree. That
is a real departure from the paper, which samples i.i.d. Gaussians; it is
recorded in ``method`` rather than hidden, and ``noise`` accepts your own
draws if you want the stochastic behaviour.

References
----------
Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A.,
Ronneberger, O. et al. (2024) "Accurate structure prediction of
biomolecular interactions with AlphaFold 3", *Nature* **630**(8016),
493-500, doi:10.1038/s41586-024-07487-w. Algorithm 18 (SampleDiffusion)
and the Table 6 defaults used below: sigma_data 16, s_max 160,
s_min 4e-4, rho 7, gamma_0 0.8, gamma_min 1.0, noise_scale 1.003,
step_scale 1.5.

Karras, T., Aittala, M., Aila, T. and Laine, S. (2022) "Elucidating the
design space of diffusion-based generative models", *NeurIPS* **35**,
26565-26577. The sigma schedule and the churn parameterisation AF3 adopts.

Vincent, P. (2011) "A connection between score matching and denoising
autoencoders", *Neural Computation* **23**(7), 1661-1674,
doi:10.1162/NECO_a_00142. The objective the fit route minimises.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["af3_sample"]

_EPS = 1e-12
# Abramson et al. (2024), supplementary Table 6
_SIGMA_DATA = 16.0
_S_MAX = 160.0
_S_MIN = 4e-4
_RHO = 7.0
_GAMMA_0 = 0.8
_GAMMA_MIN = 1.0
_NOISE_SCALE = 1.003
_STEP_SCALE = 1.5


def _atoms(x, what):
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for r in x:
        if hasattr(r, "tolist"):
            r = r.tolist()
        if not isinstance(r, (list, tuple)) or len(r) != 3:
            raise ValueError("%s: each atom needs exactly x, y, z" % what)
        out.append([float(v) for v in r])
    if not out:
        raise ValueError("%s: no atoms" % what)
    return out


def _schedule(T, sigma_data, s_max, s_min, rho):
    """Karras sigmas, descending, with a final exact zero."""
    a = s_max ** (1.0 / rho)
    b = s_min ** (1.0 / rho)
    out = []
    for i in range(int(T)):
        u = a + (i / float(T)) * (b - a)
        out.append(sigma_data * (u ** rho))
    out.append(0.0)
    return out


def _centre(X):
    n = len(X)
    c = [sum(p[a] for p in X) / n for a in range(3)]
    return [[p[a] - c[a] for a in range(3)] for p in X]


def _fit_linear_denoiser(clean, sigmas, draws, ridge=1e-6):
    r"""Denoising score matching, solved exactly.

    At each noise level the target is the clean structure and the input is
    the same structure plus noise of that scale, so the optimal linear map
    is a ridge regression and has a closed form. Solving it directly beats
    descending to it: the objective is quadratic, and an iterative solver
    would only add a tolerance to disagree about.
    """
    n = len(clean[0])
    coefs = []
    di = 0
    for s in sigmas:
        num = den = 0.0
        for X in clean:
            for i in range(n):
                for a in range(3):
                    xi = draws[di % len(draws)]
                    di += 1
                    noisy = X[i][a] + s * xi
                    num += noisy * X[i][a]
                    den += noisy * noisy
        coefs.append(num / (den + ridge) if den > _EPS else 1.0)
    return coefs


def af3_sample(n_atoms=None, denoiser=None, clean=None, steps=20,
               sigma_data=_SIGMA_DATA, s_max=_S_MAX, s_min=_S_MIN, rho=_RHO,
               gamma_0=_GAMMA_0, gamma_min=_GAMMA_MIN,
               noise_scale=_NOISE_SCALE, step_scale=_STEP_SCALE,
               noise=None, seed=2, x_init=None, ridge=1e-6):
    r"""Run the AlphaFold-3 diffusion sampler.

    Parameters
    ----------
    n_atoms : int, optional
        Number of atoms to sample. Inferred from ``x_init`` or ``clean``.
    denoiser : callable, optional
        ``denoiser(x, sigma) -> x_clean``. The trained network goes here.
    clean : list of (n, 3) arrays, optional
        Reference structures. Supplying them FITS a linear denoiser by
        denoising score matching and uses it; the coefficients come back.
    steps : int
        Schedule length T.
    sigma_data : float or "fit"
        ``"fit"`` estimates it as the RMS coordinate of ``clean`` rather
        than assuming the paper's 16, which is only right in angstroms.
    noise : array-like, optional
        Your own standard normal draws. Without it a deterministic
        low-discrepancy normal stream is used so runs reproduce exactly.

    Returns
    -------
    RichResult
        ``coords``, the ``sigmas`` walked, the per-step ``trace``, the
        fitted ``denoiser_coefs`` if any, and ``route``.
    """
    if x_init is not None:
        X0 = _atoms(x_init, "alfbnp x_init")
        n = len(X0)
    elif clean is not None:
        ref = [_atoms(c, "alfbnp clean") for c in clean]
        n = len(ref[0])
        for c in ref:
            if len(c) != n:
                raise ValueError("alfbnp: the reference structures have "
                                 "different atom counts")
        X0 = None
    elif n_atoms is not None:
        n = int(n_atoms)
        X0 = None
    else:
        raise ValueError("alfbnp: give n_atoms, x_init, or clean so the "
                         "number of atoms is known")
    if n < 1:
        raise ValueError("alfbnp: need at least one atom")
    T = int(steps)
    if T < 1:
        raise ValueError("alfbnp: steps must be at least 1")

    if clean is not None:
        ref = [_atoms(c, "alfbnp clean") for c in clean]
    else:
        ref = None

    if isinstance(sigma_data, str):
        if sigma_data != "fit":
            raise ValueError("alfbnp: sigma_data must be a number or 'fit'")
        if ref is None:
            raise ValueError("alfbnp: sigma_data='fit' needs `clean`")
        tot = cnt = 0.0
        for X in ref:
            for p in X:
                for a in range(3):
                    tot += p[a] * p[a]
                    cnt += 1.0
        sd = math.sqrt(tot / cnt) if cnt else _SIGMA_DATA
    else:
        sd = float(sigma_data)
    if not sd > 0.0:
        raise ValueError("alfbnp: sigma_data must be positive")

    sig = _schedule(T, sd, float(s_max), float(s_min), float(rho))

    need = 3 * n * (T + 2) + (3 * n * len(ref) * (T + 1) if ref else 0)
    if noise is not None:
        z = [float(v) for v in
             (noise.tolist() if hasattr(noise, "tolist") else noise)]
        if not z:
            raise ValueError("alfbnp: noise is empty")
    else:
        z = k.normdraws(max(need, 8), int(seed))
    zi = 0

    coefs = None
    if denoiser is None:
        if ref is None:
            raise ValueError(
                "alfbnp: no denoiser and no `clean` to fit one from. The "
                "network is not bundled and will not be invented: supply a "
                "trained denoiser, or reference structures to fit a linear "
                "one by denoising score matching.")
        coefs = _fit_linear_denoiser(ref, sig[:-1], z, ridge=float(ridge))
        route = "fitted a linear denoiser by denoising score matching"

        def denoise(x, s, _i=[0]):
            c = coefs[min(_i[0], len(coefs) - 1)]
            _i[0] += 1
            return [[c * v for v in p] for p in x]
    else:
        route = "sampled with a supplied denoiser"

        def denoise(x, s):
            return _atoms(denoiser(x, s), "alfbnp denoiser output")

    # x <- sigma(t_0) * N(0, I)
    if X0 is not None:
        X = [list(p) for p in X0]
    else:
        X = []
        for _ in range(n):
            X.append([sig[0] * z[(zi + a) % len(z)] for a in range(3)])
            zi += 3

    trace = []
    for i in range(1, T + 1):
        X = _centre(X)                       # CentreRandomAugmentation, the
        # rotation part is omitted deliberately: it is a training-time
        # augmentation and applying it at sampling only adds an arbitrary
        # frame, which would make the two arms disagree for no gain.
        prev = sig[i - 1]
        gamma = float(gamma_0) if sig[i] > float(gamma_min) else 0.0
        t_hat = prev * (gamma + 1.0)
        var = t_hat * t_hat - prev * prev
        step_noise = (float(noise_scale) * math.sqrt(var)
                      if var > 0.0 else 0.0)
        Xn = []
        for p in X:
            row = []
            for a in range(3):
                row.append(p[a] + step_noise * z[zi % len(z)])
                zi += 1
            Xn.append(row)
        Xd = denoise(Xn, t_hat)
        if len(Xd) != n:
            raise ValueError("alfbnp: the denoiser returned %d atoms, not %d"
                             % (len(Xd), n))
        dt = sig[i] - t_hat
        X = [[Xn[j][a] + float(step_scale) * dt
              * ((Xn[j][a] - Xd[j][a]) / t_hat if t_hat > _EPS else 0.0)
              for a in range(3)] for j in range(n)]
        rms = math.sqrt(sum(v * v for p in X for v in p) / (3 * n))
        trace.append([i, sig[i], t_hat, rms])

    rmsd_to_ref = None
    if ref is not None:
        best = None
        for R in ref:
            d = math.sqrt(sum((X[j][a] - R[j][a]) ** 2
                              for j in range(n) for a in range(3)) / n)
            if best is None or d < best:
                best = d
        rmsd_to_ref = best

    return RichResult(payload={
        "estimate": rmsd_to_ref if rmsd_to_ref is not None else sig[-1],
        "coords": X,
        "sigmas": sig,
        "trace": trace,
        "denoiser_coefs": coefs,
        "sigma_data": sd,
        "steps": T,
        "rmsd_to_reference": rmsd_to_ref,
        "n_atoms": n,
        "route": route,
        "method": ("AlphaFold-3 SampleDiffusion (Abramson et al. 2024, "
                   "Algorithm 18) on the Karras sigma schedule, with the "
                   "Table 6 defaults; the noise stream is the package's "
                   "deterministic low-discrepancy normal sequence rather "
                   "than i.i.d. draws, so a run reproduces exactly"),
        "note": ("route says whether the denoiser was supplied or fitted. "
                 "The network is not bundled. The rotation half of "
                 "CentreRandomAugmentation is deliberately not applied: it "
                 "is a training-time augmentation, and at sampling it only "
                 "chooses an arbitrary frame, which both costs "
                 "reproducibility and gains nothing. Pass `noise` to "
                 "recover genuinely stochastic sampling."),
    })


def cheatsheet():
    return ("alfbnp: af3_sample(n_atoms, denoiser=...) or af3_sample("
            "clean=[...]) -> AlphaFold-3 diffusion sampling (Abramson et "
            "al. 2024 Nature 630:493, Algorithm 18)")


# compact alias per ledger/NAMING.md
af3_protein_ligand = af3_sample
