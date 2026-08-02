# morie.fn -- internal helpers (rootcoder007/morie)
"""Shared machinery for the transformation model T(Y) = X'beta + U.

Spec: Horowitz, J. L., *Semiparametric and Nonparametric Methods in
Econometrics*, Springer, Sec. 6.3. Equation numbers below are the
book's.

The model is identified only up to location and scale, and the two
normalisations used throughout are the book's own (p. 215):

* SCALE: ``|beta_1| = 1``. Replacing T by cT, beta by c*beta and U by
  cU leaves the model unchanged for any c > 0.
* LOCATION: ``T(y0) = 0`` at a chosen y0. Replacing T by T + d and U
  by U + d leaves it unchanged for any d.

With the location normalisation there is NO centering assumption on U
and NO intercept in X. Estimates produced under a different
normalisation are not comparable without the adjustment on p. 216.
"""

from . import _array_core as np

__all__ = ["normalize_scale", "kernel_K", "kernel_Kz_sixth",
           "kernel_Kz_sixth_deriv", "SCALE_NOTE"]

SCALE_NOTE = "|beta_1| = 1 (scale) and T(y0) = 0 (location); no intercept in X"


def normalize_scale(beta):
    """Impose |beta_1| = 1 by dividing through by |beta_1|."""
    b = np.asarray(beta, dtype=float).ravel()
    if b.size == 0:
        raise ValueError("beta must be non-empty.")
    if b[0] == 0:
        raise ValueError(
            "the first component of beta is zero, so |beta_1| = 1 cannot be "
            "imposed; reorder X so a component with a nonzero coefficient "
            "and a continuous conditional distribution comes first.")
    return b / abs(b[0])


def kernel_K(u):
    """Second-order Gaussian kernel, used for K_Y."""
    u = np.asarray(u, dtype=float)
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def kernel_Kz_sixth(u):
    r"""Sixth-order kernel for K_Z, required by assumption HT8.

    HT8 asks K_Z to satisfy :math:`\int v^j K_Z(v)dv = 0` for
    ``1 <= j <= r-1`` with ``r >= 6``. A second-order kernel will NOT
    do: :math:`G_{nz}` is a functional of DERIVATIVES of K_Z, those
    converge relatively slowly, and the higher-order kernel is what
    restores fast enough convergence (p. 220-221). The Gaussian-based
    sixth-order kernel is

    .. math:: K(u) = \frac{1}{16}(15 - 10u^2 + u^4)\phi(u),

    whose second and fourth moments vanish by construction.
    """
    u = np.asarray(u, dtype=float)
    return (15.0 - 10.0 * u**2 + u**4) / 16.0 * kernel_K(u)


def kernel_Kz_sixth_deriv(u):
    r"""Derivative of :func:`kernel_Kz_sixth`.

    .. math:: K'(u) = \frac{\phi(u)}{16}\big(-35u + 14u^3 - u^5\big),

    obtained by differentiating
    :math:`(15 - 10u^2 + u^4)\phi(u)/16` and using
    :math:`\phi'(u) = -u\phi(u)`. G_nz is a derivative functional of
    K_Z, so this is needed explicitly rather than by differencing.
    """
    u = np.asarray(u, dtype=float)
    return kernel_K(u) * (-35.0 * u + 14.0 * u**3 - u**5) / 16.0


def cheatsheet():
    return "_hrz_transform: |beta_1|=1 and T(y0)=0; K_Z must be sixth order, not second"
