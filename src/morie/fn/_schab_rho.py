"""Valid parameter space for spatial autoregressive dependence parameters.

A SAR/CAR/Durbin likelihood only exists where the implied precision (or
|I - rho W|) is positive definite. The bound is an eigenvalue condition:

    rho in (1 / theta_min, 1 / theta_max)

with theta the eigenvalues of the relevant matrix. Schabenberger &
Gotway state it this way in eq. (6.48), p. 340, where "theta_i are the
eigenvalues of W".

Hardcoding an interval instead -- (-0.99, 0.99) is the common choice --
is only safe when W happens to be row-standardised AND the true bound is
wider. For a raw adjacency it is not: on a 24-node chain the valid range
is (-0.504, 0.504), so a (-0.99, 0.99) search spends most of its time
where the likelihood is undefined.
"""

import numpy as np

__all__ = []


def rho_bounds(W, form="identity"):
    """Open interval of admissible rho.

    ``identity``  -- for ``I - rho W`` (SAR, Durbin, CAR with Sigma_c =
    sigma^2 I): eigenvalues of the symmetrised W.

    ``weighted``  -- for ``D - rho W`` (Besag CAR): eigenvalues of
    ``D^-1/2 W D^-1/2``.

    Returns ``(lo, hi)``; either end is infinite when the corresponding
    eigenvalue does not change sign.
    """
    W = np.asarray(W, dtype=float)
    if form == "weighted":
        d = W.sum(axis=1)
        s = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
        M = (W * s[:, None]) * s[None, :]
    elif form == "identity":
        M = W
    else:
        raise ValueError("`form` must be 'identity' or 'weighted'")
    ev = np.linalg.eigvalsh((M + M.T) / 2.0)
    lo = 1.0 / ev.min() if ev.min() < 0 else -np.inf
    hi = 1.0 / ev.max() if ev.max() > 0 else np.inf
    return float(lo), float(hi)


def safe_search_interval(W, form="identity", pad=1e-6):
    """``rho_bounds`` shrunk by ``pad`` and clipped to something finite.

    Optimisers need a closed interval; the bound itself is open, and the
    likelihood diverges at the ends.
    """
    lo, hi = rho_bounds(W, form)
    lo = max(lo, -1e6) if np.isfinite(lo) else -1e6
    hi = min(hi, 1e6) if np.isfinite(hi) else 1e6
    eps = pad * max(hi - lo, 1e-12)
    return lo + eps, hi - eps
