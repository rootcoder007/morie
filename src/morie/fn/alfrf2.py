"""RFdiffusion: designing a protein backbone by denoising, with a
functional motif held fixed throughout.

The design problem is inverse. A binding site, an active site, a metal
coordination sphere -- some small set of residues whose geometry is the
whole point -- is known, and what is missing is a protein that holds
those residues in exactly that arrangement. RFdiffusion answers it by
running a denoising diffusion model over backbone coordinates: start
from noise, walk the noise back down to a structure, and at every step
overwrite the motif positions with the motif, correctly noised for the
level you are at. The scaffold is whatever the denoiser grows around
the part that was never allowed to move.

The pieces, and which of them are arithmetic rather than weights:

  THE SCHEDULE. A linear variance schedule and its cumulative products,
  from Ho, Jain and Abbeel. The closed form for the forward process,
  x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, and the closed form
  for the reverse posterior mean given a predicted x_0, are both exact
  and both here.

  THE MOTIF CONSTRAINT. At every reverse step the motif rows are
  replaced by the motif noised to that step's level -- the inpainting
  rule of Lugmayr et al., which is what makes the motif a hard
  constraint rather than a suggestion. At the last step the noise level
  is zero, so the motif comes out EXACTLY where it was put in. That is
  an equality, not a tolerance, and it is anchored as one.

  THE DENOISER. This is the part that is a trained network in the
  paper, and there are no weights here. Two routes stand in for it,
  both named for what they are:

    ``prior``  predicts the prior mean, which is zero. The reverse
               process then samples the prior, conditioned on the
               motif. It is the correct baseline and it is not protein
               design; it is here so the effect of a real denoiser can
               be measured against something.

    ``ideal``  relaxes consecutive alpha carbons toward the 3.8 angstrom
               spacing that a trans peptide bond forces on a backbone,
               with the motif pinned. That distance is a fact of
               covalent geometry, not a fitted parameter, so the route
               is honest -- but it knows nothing about sequence,
               packing or secondary structure, and it is not a
               substitute for the network.

  A caller with a real denoiser passes it as ``denoiser`` and gets the
  real reverse process; the routes above are then irrelevant.

  THE MEASUREMENT. Motif RMSD after optimal superposition, by Kabsch's
  construction, computed from the eigendecomposition of the covariance
  in the shared Jacobi routine rather than a second copy of it. RMSD
  against a rigidly moved copy of a structure is exactly zero, which is
  the anchor that proves the superposition is doing its job.

References
  Watson, J.L., Juergens, D., Bennett, N.R., Trippe, B.L., Yim, J.,
    Eisenach, H.E., Ahern, W., Borst, A.J., Ragotte, R.J., Milles,
    L.F., Wicky, B.I.M., Hanikel, N., Pellock, S.J., Courbet, A.,
    Sheffler, W., Wang, J., Venkatesh, P., Sappington, I., Torres,
    S.V., Lauko, A., De Bortoli, V., Mathieu, E., Ovchinnikov, S.,
    Barzilay, R., Jaakkola, T.S., DiMaio, F., Baek, M. and Baker, D.
    (2023) "De novo design of protein structure and function with
    RFdiffusion." Nature 620, 1089-1100.
    doi:10.1038/s41586-023-06415-8.
  Ho, J., Jain, A. and Abbeel, P. (2020) "Denoising diffusion
    probabilistic models." Advances in Neural Information Processing
    Systems 33, 6840-6851. The schedule, the forward closed form and
    the reverse posterior used here.
  Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R. and Van
    Gool, L. (2022) "RePaint: inpainting using denoising diffusion
    probabilistic models." Proceedings of the IEEE/CVF Conference on
    Computer Vision and Pattern Recognition, 11461-11471. The
    replace-the-known-region rule that makes the motif a constraint.
  Kabsch, W. (1976) "A solution for the best rotation to relate two
    sets of vectors." Acta Crystallographica A32(5), 922-923.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from .manfd import jacobi_eigen
from ._richresult import RichResult

__all__ = ["rfdiffusion_protein", "ddpm_schedule", "forward_noise",
           "kabsch", "rmsd", "ideal_chain", "cheatsheet"]

# The alpha carbon separation a trans peptide bond forces on successive
# residues. A fact of covalent geometry, not a fitted constant.
CA_SPACING = 3.8


def ddpm_schedule(T, beta_start=1e-4, beta_end=0.02):
    """The linear variance schedule and its cumulative products.

    Returns betas, alphas and alpha-bars, indexed from step one, with
    ``abar[0] = 1`` standing for the clean structure so the forward and
    reverse formulas can be written without a special case at the ends.
    The endpoints are the ones Ho et al. use; RFdiffusion's own schedule
    is a parameter here rather than a constant, because it is a training
    choice and not a property of diffusion.
    """
    T = int(T)
    if T < 1:
        raise ValueError("a diffusion needs at least one step")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("the variance schedule must rise through the "
                         "open unit interval")
    betas = [0.0] * (T + 1)
    alphas = [1.0] * (T + 1)
    abar = [1.0] * (T + 1)
    for t in range(1, T + 1):
        b = beta_start + (beta_end - beta_start) * (t - 1) / max(T - 1, 1)
        betas[t] = b
        alphas[t] = 1.0 - b
        abar[t] = abar[t - 1] * (1.0 - b)
    return betas, alphas, abar


def forward_noise(x0, abar_t, eps):
    """The forward process in closed form: no loop over steps.

    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps. At abar equal to one
    this is the structure itself, which is what makes the motif exact at
    the end of the reverse walk.
    """
    a = math.sqrt(abar_t)
    b = math.sqrt(1.0 - abar_t)
    return [[a * x0[i][d] + b * eps[i][d] for d in range(len(x0[i]))]
            for i in range(len(x0))]


def _centre(P):
    n = len(P)
    c = [_w.csum(P[i][d] for i in range(n)) / n for d in range(3)]
    return [[P[i][d] - c[d] for d in range(3)] for i in range(n)], c


def _det3(M):
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def kabsch(P, Q):
    """The rigid motion that best takes P onto Q, and the residual.

    The rotation is the orthogonal polar factor of the cross-covariance
    C = sum q p', obtained as C (C'C)^{-1/2} from the eigendecomposition
    of C'C. If that factor comes out with a negative determinant it is a
    reflection, not a rotation, and the sign is flipped on the direction
    of least variance -- the standard correction, and the reason a naive
    superposition can silently report a mirror image as a match.
    """
    n = len(P)
    if n != len(Q):
        raise ValueError("superposition needs the same number of points "
                         "on both sides")
    if n < 3:
        raise ValueError("three points are the fewest that fix a rotation")
    p, cp = _centre(P)
    q, cq = _centre(Q)
    C = [[_w.csum(q[i][a] * p[i][b] for i in range(n)) for b in range(3)]
         for a in range(3)]
    S = [[_w.csum(C[k][a] * C[k][b] for k in range(3)) for b in range(3)]
         for a in range(3)]
    lam, V = jacobi_eigen(S)
    if lam[2] <= 1e-12 * (lam[0] if lam[0] > 0.0 else 1.0):
        raise ValueError("the points do not span three dimensions, so "
                         "the polar factor does not determine a "
                         "rotation")
    inv = [1.0 / math.sqrt(v) for v in lam]
    M = [[_w.csum(V[a][k] * inv[k] * V[b][k] for k in range(3))
          for b in range(3)] for a in range(3)]
    R = [[_w.csum(C[a][k] * M[k][b] for k in range(3)) for b in range(3)]
         for a in range(3)]
    if _det3(R) < 0.0:
        inv[2] = -inv[2]
        M = [[_w.csum(V[a][k] * inv[k] * V[b][k] for k in range(3))
              for b in range(3)] for a in range(3)]
        R = [[_w.csum(C[a][k] * M[k][b] for k in range(3))
              for b in range(3)] for a in range(3)]
    moved = [[_w.csum(R[a][b] * p[i][b] for b in range(3)) + cq[a]
              for a in range(3)] for i in range(n)]
    sq = _w.csum((moved[i][a] - Q[i][a]) * (moved[i][a] - Q[i][a])
                 for i in range(n) for a in range(3))
    return R, [cq[a] - _w.csum(R[a][b] * cp[b] for b in range(3))
               for a in range(3)], math.sqrt(sq / n), moved


def rmsd(P, Q):
    """Root-mean-square deviation after optimal superposition."""
    return kabsch(P, Q)[2]


def ideal_chain(x, fixed, spacing=CA_SPACING, passes=8):
    """Relax consecutive alpha carbons toward the backbone spacing.

    A symmetric constraint relaxation: each consecutive pair is moved
    along the line joining it until the pair is the right distance
    apart, both ends sharing the correction equally unless one of them
    is pinned, in which case the free end takes all of it. Repeated for
    a fixed number of passes so the result does not depend on a
    convergence test.

    Two points on top of each other have no direction to be separated
    along, so that pair is left alone rather than moved somewhere
    arbitrary.
    """
    y = [list(p) for p in x]
    n = len(y)
    for _ in range(int(passes)):
        for i in range(n - 1):
            j = i + 1
            d = [y[j][a] - y[i][a] for a in range(3)]
            L = math.sqrt(_w.csum(v * v for v in d))
            if L == 0.0:
                continue
            corr = (L - spacing) / L
            fi = i in fixed
            fj = j in fixed
            if fi and fj:
                continue
            wi = 0.0 if fi else (1.0 if fj else 0.5)
            wj = 0.0 if fj else (1.0 if fi else 0.5)
            for a in range(3):
                y[i][a] = y[i][a] + wi * corr * d[a]
                y[j][a] = y[j][a] - wj * corr * d[a]
    return y


def _denoise(route, denoiser, x, t, fixed, spacing, passes):
    if denoiser is not None:
        return [[float(v) for v in row] for row in denoiser(x, t)]
    if route == "prior":
        return [[0.0, 0.0, 0.0] for _ in x]
    if route == "ideal":
        return ideal_chain(x, fixed, spacing, passes)
    raise ValueError("the denoiser route is prior or ideal")


def rfdiffusion_protein(target_motif, scaffold, T=20, denoise="ideal",
                        denoiser=None, beta_start=1e-4, beta_end=0.02,
                        spacing=CA_SPACING, passes=8, noise_scale=1.0,
                        seed=0):
    """Grow a backbone around a fixed motif by reverse diffusion.

    Parameters
    ----------
    target_motif : sequence
        Pairs of a residue index and the three coordinates that residue
        must end up at.
    scaffold : int or sequence
        The number of residues in the design, or a starting structure
        to noise from.
    T : int
        Diffusion steps.
    denoise : {"ideal", "prior"}
        Which stand-in denoiser to use; see the module docstring on
        what each is and is not.
    denoiser : callable or None
        ``f(x, t)`` returning the predicted clean structure. Given, it
        replaces the routes entirely.
    noise_scale : float
        A multiplier on the reverse-process noise. One is the sampler
        as published; zero makes the walk deterministic, which is what
        the reproducibility anchor uses.

    Returns
    -------
    RichResult
        The designed backbone, the motif it was built around, the motif
        RMSD -- which must be zero -- and the chain geometry it came out
        with.

    References
    ----------
    Watson et al. (2023) Nature 620, 1089-1100; Ho et al. (2020) NeurIPS
    33, 6840-6851; Lugmayr et al. (2022) CVPR, 11461-11471.
    """
    motif = [(int(i), [float(v) for v in xyz]) for i, xyz in target_motif]
    if isinstance(scaffold, int):
        n = scaffold
        start = None
    else:
        start = [[float(v) for v in row] for row in scaffold]
        n = len(start)
    if n < 3:
        raise ValueError("a backbone of fewer than three residues has no "
                         "geometry to design")
    for i, _ in motif:
        if i < 0 or i >= n:
            raise ValueError("a motif residue falls outside the design")
    if len(set(i for i, _ in motif)) != len(motif):
        raise ValueError("a residue cannot be pinned to two places")
    fixed = set(i for i, _ in motif)

    betas, alphas, abar = ddpm_schedule(T, beta_start, beta_end)
    rng = _core._SplitMix64(seed)

    # The motif written into a full-length frame, so it can be noised
    # with the same closed form as everything else.
    m0 = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i, xyz in motif:
        m0[i] = list(xyz)

    if start is None:
        x = [[rng.normal() for _ in range(3)] for _ in range(n)]
    else:
        eps = [[rng.normal() for _ in range(3)] for _ in range(n)]
        x = forward_noise(start, abar[T], eps)
    e = [[rng.normal() for _ in range(3)] for _ in range(n)]
    mt = forward_noise(m0, abar[T], e)
    for i in fixed:
        x[i] = list(mt[i])

    traj = []
    for t in range(T, 0, -1):
        x0 = _denoise(denoise, denoiser, x, t, fixed, spacing, passes)
        for i, xyz in motif:
            x0[i] = list(xyz)
        c1 = math.sqrt(abar[t - 1]) * betas[t] / (1.0 - abar[t])
        c2 = (math.sqrt(alphas[t]) * (1.0 - abar[t - 1])
              / (1.0 - abar[t]))
        sd = math.sqrt(betas[t] * (1.0 - abar[t - 1]) / (1.0 - abar[t]))
        z = [[rng.normal() for _ in range(3)] for _ in range(n)]
        nxt = [[c1 * x0[i][d] + c2 * x[i][d]
                + (noise_scale * sd * z[i][d] if t > 1 else 0.0)
                for d in range(3)] for i in range(n)]
        # The known region is replaced by the motif noised to the level
        # we have arrived at. At t equal to one that level is zero, so
        # the motif lands exactly.
        e = [[rng.normal() for _ in range(3)] for _ in range(n)]
        mt = forward_noise(m0, abar[t - 1], e)
        for i in fixed:
            nxt[i] = list(mt[i])
        x = nxt
        traj.append(_w.csum(x[i][d] * x[i][d]
                            for i in range(n) for d in range(3)))

    got = [x[i] for i, _ in motif]
    want = [xyz for _, xyz in motif]
    mdev = max(abs(got[k][d] - want[k][d])
               for k in range(len(motif)) for d in range(3)) \
        if motif else 0.0
    spac = [math.sqrt(_w.csum((x[i + 1][d] - x[i][d])
                              * (x[i + 1][d] - x[i][d])
                              for d in range(3))) for i in range(n - 1)]
    # Three or fewer motif residues, or coplanar ones, do not pin a
    # rotation, and the superposition says so rather than returning a
    # number it cannot justify.
    try:
        mr = rmsd(got, want) if len(motif) >= 3 else 0.0
    except ValueError:
        mr = float("nan")
    cen = [_w.csum(x[i][d] for i in range(n)) / n for d in range(3)]
    rg = math.sqrt(_w.csum((x[i][d] - cen[d]) * (x[i][d] - cen[d])
                           for i in range(n) for d in range(3)) / n)
    return RichResult(payload={
        "backbone": x,
        "motif_index": [i for i, _ in motif],
        "motif_target": want,
        "motif_placed": got,
        "motif_max_deviation": mdev,
        "motif_rmsd": mr,
        "spacing": spac,
        "mean_spacing": (_w.csum(spac) / len(spac)) if spac else 0.0,
        "radius_of_gyration": rg,
        "trace": traj,
        "n": n,
        "n_motif": len(motif),
        "T": int(T),
        "denoise": denoise if denoiser is None else "callable",
        "noise_scale": float(noise_scale),
        "seed": seed,
        "method": "RFdiffusion motif-scaffolding reverse diffusion",
    })


def cheatsheet():
    return ("alfrf2: RFdiffusion motif scaffolding. Reverse DDPM over "
            "backbone coordinates with the motif replaced at every step, "
            "so it lands exactly; denoiser routes are prior or ideal")
