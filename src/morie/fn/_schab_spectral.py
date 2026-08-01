# SPDX-License-Identifier: AGPL-3.0-or-later
"""The periodogram on a rectangular lattice.

Schabenberger & Gotway (2005), Sec. 4.7.1, pp. 190-192.

  eq (4.56)  s(w1,w2) = (2 pi)^-2 sum_j sum_k C(j,k) exp{-i(w1 j + w2 k)}
  eq (4.57)  I(w1,w2) = 1/{(2 pi)^2 r c}
                        | sum_u sum_v Z(u,v) exp{-i(w1 u + w2 v)} |^2
  eq (4.58)  2 r pi I(w_j) = sum_u sum_p (Z(u)-Zbar)(Z(p)-Zbar) cos(w_j(u-p))
  eq (4.59)  I(w1,w2) = (2 pi)^-2 sum_j sum_k Chat(j,k) cos{w1 j + w2 k}

Fourier frequencies (p. 190): multiples of 2 pi / r and 2 pi / c,

  w1 = (2 pi / r) j,  j = -floor((r-1)/2), ..., floor(r/2)
  w2 = (2 pi / c) k,  k = -floor((c-1)/2), ..., floor(c/2)

The stub this module replaces printed ``I(omega) = (1/n)|sum Z(s)
exp(-i omega' s)|^2``, dropping the ``(2 pi)^2`` of eq (4.57). That constant
is not cosmetic: eq (4.59) states the periodogram IS the Fourier transform of
the sample covariance function, and the identity only closes with the
``(2 pi)^-2`` in place. :func:`periodogram_from_covariance` computes the
right-hand side of (4.59) independently so the two can be compared; they
agree to machine precision, which is what pins the normalisation.

The book is explicit that (4.59) holds "for w1 = 0 and w2 = 0" -- read in
context (p. 192) as the non-zero frequencies, since at the origin the
periodogram carries the squared mean while the sample covariances are
mean-centred. ``omit_zero_frequency`` defaults to True for that reason and
the discrepancy at the origin is reported rather than hidden.
"""

import numpy as np

__all__ = [
    "fourier_frequencies",
    "periodogram",
    "periodogram_from_covariance",
    "sample_covariance_2d",
    "spectral_density",
]


def _as_lattice(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"a rectangular r x c lattice is required, got ndim={z.ndim}")
    if z.shape[0] < 2 or z.shape[1] < 2:
        raise ValueError(f"lattice must be at least 2x2, got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("lattice contains non-finite values")
    return z


def fourier_frequencies(r, c):
    """The Fourier frequency grid of p. 190.

    Returns ``(w1, w2, j, k)`` where ``j`` and ``k`` are the integer
    multipliers, running from ``-floor((n-1)/2)`` to ``floor(n/2)`` so that
    exactly ``n`` frequencies are produced in each direction.
    """
    r, c = int(r), int(c)
    if r < 2 or c < 2:
        raise ValueError("r and c must both be at least 2")
    j = np.arange(-((r - 1) // 2), r // 2 + 1)
    k = np.arange(-((c - 1) // 2), c // 2 + 1)
    return 2.0 * np.pi * j / r, 2.0 * np.pi * k / c, j, k


def sample_covariance_2d(z, centre=True):
    """Sample covariance ``Chat(j,k)`` for lags ``j in [-(r-1), r-1]``,
    ``k in [-(c-1), c-1]``.

    Follows the one-dimensional definition on p. 191 -- the divisor is the
    full lattice size ``r*c``, not the number of overlapping pairs, which is
    what makes (4.59) close exactly.
    """
    z = _as_lattice(z)
    r, c = z.shape
    d = z - z.mean() if centre else z
    lags_j = np.arange(-(r - 1), r)
    lags_k = np.arange(-(c - 1), c)
    out = np.zeros((lags_j.size, lags_k.size))
    for a, jj in enumerate(lags_j):
        for b, kk in enumerate(lags_k):
            u0, u1 = max(0, -jj), min(r, r - jj)
            v0, v1 = max(0, -kk), min(c, c - kk)
            if u1 <= u0 or v1 <= v0:
                continue
            left = d[u0:u1, v0:v1]
            right = d[u0 + jj:u1 + jj, v0 + kk:v1 + kk]
            out[a, b] = float((left * right).sum()) / (r * c)
    return out, lags_j, lags_k


def periodogram(z, omit_zero_frequency=True):
    """eq (4.57): the periodogram at the Fourier frequencies.

    ``I(w1,w2) = 1/{(2 pi)^2 r c} | sum_u sum_v Z(u,v) exp{-i(w1 u + w2 v)} |^2``

    The book notes (p. 191) that ``sum_u cos(w_j u) = 0`` at a non-zero
    Fourier frequency, so the mean may be subtracted without altering the
    value; ``mean_invariant`` in the result records whether that held.
    """
    z = _as_lattice(z)
    r, c = z.shape
    w1, w2, j, k = fourier_frequencies(r, c)
    u = np.arange(1, r + 1)
    v = np.arange(1, c + 1)

    def _spec(field):
        # exp{-i(w1 u + w2 v)} summed over the lattice, for every (w1, w2)
        eu = np.exp(-1j * np.outer(w1, u))          # (nw1, r)
        ev = np.exp(-1j * np.outer(w2, v))          # (nw2, c)
        amp = eu @ field @ ev.T                     # (nw1, nw2)
        return np.abs(amp) ** 2 / ((2.0 * np.pi) ** 2 * r * c)

    inten = _spec(z)
    inten_centred = _spec(z - z.mean())

    zero_j = int(np.where(j == 0)[0][0])
    zero_k = int(np.where(k == 0)[0][0])
    mask = np.ones_like(inten, dtype=bool)
    mask[zero_j, :] = False
    mask[:, zero_k] = False
    invariant = bool(np.allclose(inten[mask], inten_centred[mask], rtol=1e-9, atol=1e-12))

    result = inten_centred if omit_zero_frequency else inten
    return {
        "periodogram": result,
        "omega1": w1,
        "omega2": w2,
        "j": j,
        "k": k,
        "zero_index": (zero_j, zero_k),
        "nonzero_mask": mask,
        "mean_invariant": invariant,
        "r": r,
        "c": c,
    }


def periodogram_from_covariance(z):
    """eq (4.59): the same quantity as the Fourier transform of ``Chat``.

    ``I(w1,w2) = (2 pi)^-2 sum_j sum_k Chat(j,k) cos{w1 j + w2 k}``

    Computed independently of :func:`periodogram` so the two can be checked
    against each other. That agreement is the section's central claim.
    """
    z = _as_lattice(z)
    r, c = z.shape
    chat, lags_j, lags_k = sample_covariance_2d(z)
    w1, w2, j, k = fourier_frequencies(r, c)
    out = np.zeros((w1.size, w2.size))
    for a, o1 in enumerate(w1):
        for b, o2 in enumerate(w2):
            ang = np.add.outer(o1 * lags_j, o2 * lags_k)
            out[a, b] = float((chat * np.cos(ang)).sum()) / ((2.0 * np.pi) ** 2)
    return {"periodogram": out, "omega1": w1, "omega2": w2, "covariance": chat,
            "lags_j": lags_j, "lags_k": lags_k}


def spectral_density(cov, lags_j, lags_k, omega1, omega2):
    """eq (4.56): the spectral density of a covariance function.

    ``s(w1,w2) = (2 pi)^-2 sum_j sum_k C(j,k) cos{w1 j + w2 k}``

    The book gives the exponential and cosine forms as equal, which holds
    because a valid covariance function is even in its lags.
    """
    cov = np.asarray(cov, dtype=float)
    lags_j = np.asarray(lags_j, dtype=float)
    lags_k = np.asarray(lags_k, dtype=float)
    if cov.shape != (lags_j.size, lags_k.size):
        raise ValueError(
            f"cov is {cov.shape} but lags are {lags_j.size} x {lags_k.size}")
    ang = np.add.outer(float(omega1) * lags_j, float(omega2) * lags_k)
    return float((cov * np.cos(ang)).sum()) / ((2.0 * np.pi) ** 2)
