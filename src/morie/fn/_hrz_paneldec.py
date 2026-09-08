# morie.fn -- internal helpers (rootcoder007/morie)
"""Panel-data deconvolution for Y_jt = X_jt'beta + U_j + eps_jt.

Spec: Horowitz, *Semiparametric and Nonparametric Methods in
Econometrics*, Sec. 5.2. Equation numbers are the book's.

The identification trick and its precondition:

* (5.21) ``W_njt = Y_jt - b_n'X_jt`` estimates ``W = U + eps``, so
  ``psi_W = psi_U * psi_eps``.
* (5.22) ``eta_njt = (Y_jt - Y_j1) - b_n'(X_jt - X_j1)`` estimates
  the DIFFERENCE of two independent copies of eps, which removes U
  entirely, so ``psi_eta = |psi_eps|^2``.
* Therefore ``psi_eps = psi_eta^(1/2)`` and
  ``psi_U = psi_W / psi_eta^(1/2)``, both positive roots.

That square root is only legitimate because eps is assumed
SYMMETRIC about zero, which makes ``psi_eps`` real, and because
``psi_eps(tau) != 0`` for all finite tau, which with continuity and
``psi_eps(0) = 1`` forces ``psi_eps > 0``. Drop symmetry and the
sign of the root is not identified. The book is explicit that
symmetry is not required for identification in general, but it IS
what this particular construction rests on (Horowitz and Markatou
1996 show how to relax it).
"""

from . import _array_core as np

__all__ = ["panel_residuals", "char_funcs", "smoothing_cf", "deconvolve_pair"]


def panel_residuals(y, x, beta):
    """(5.21) and (5.22) from panel arrays shaped (n, T) and (n, T, d)."""
    Y = np.asarray(y, dtype=float)
    X = np.asarray(x, dtype=float)
    b = np.asarray(beta, dtype=float).ravel()
    if Y.ndim != 2:
        raise ValueError(f"y must be (n, T), got shape {Y.shape}.")
    n, T = Y.shape
    if T < 2:
        raise ValueError(f"need at least 2 periods, got {T}.")
    if X.ndim == 2:
        if X.shape == (n * T, b.size):
            X = X.reshape(n, T, b.size)
        elif X.shape == (n, T):
            X = X[..., None]
        else:
            raise ValueError(f"x has shape {X.shape}, cannot match {(n, T)}.")
    if X.shape[:2] != (n, T):
        raise ValueError(f"x has shape {X.shape}, expected {(n, T)} plus d.")
    if X.shape[2] != b.size:
        raise ValueError(
            f"beta has {b.size} entries for {X.shape[2]} covariates.")
    W = Y - X @ b                                                    # (5.21)
    eta = (Y[:, 1:] - Y[:, :1]) - (X[:, 1:, :] - X[:, :1, :]) @ b    # (5.22)
    return W.ravel(), eta.ravel()


def char_funcs(W, eta, tau):
    """Empirical characteristic functions of W and eta."""
    psi_W = np.exp(1j * np.outer(tau, W)).mean(axis=1)
    psi_eta = np.exp(1j * np.outer(tau, eta)).mean(axis=1)
    return psi_W, psi_eta


def smoothing_cf(u):
    r"""``psi_zeta``: a bounded real characteristic function supported
    on [-1, 1].

    The book's own example is the fourfold convolution of the uniform
    density with itself, whose characteristic function is
    :math:`\operatorname{sinc}^4`. Compact support in tau is the
    whole point: it is what stops the integrand being evaluated where
    the denominator vanishes.
    """
    u = np.asarray(u, dtype=float)
    s = np.sinc(u / (4.0 * np.pi)) ** 4
    return np.where(np.abs(u) <= 1.0, s, 0.0)


def deconvolve_pair(W, eta, grid_u, grid_z, nu_U, nu_eps, n_tau=2001):
    """(5.25) and (5.26): smoothed estimators of f_eps and f_U."""
    nu_U = float(nu_U)
    nu_eps = float(nu_eps)
    if nu_U <= 0 or nu_eps <= 0:
        raise ValueError(f"bandwidths must be positive, got {(nu_U, nu_eps)}.")
    tau_u = np.linspace(-1.0 / nu_U, 1.0 / nu_U, int(n_tau))
    tau_e = np.linspace(-1.0 / nu_eps, 1.0 / nu_eps, int(n_tau))

    _, psi_eta_e = char_funcs(W, eta, tau_e)
    # (5.25)
    integ_e = np.sqrt(np.abs(psi_eta_e)) * smoothing_cf(nu_eps * tau_e)
    f_eps = np.array([
        float(np.real(np.trapezoid(integ_e * np.exp(-1j * tau_e * z), tau_e))
              / (2 * np.pi)) for z in np.atleast_1d(grid_z)])

    psi_W_u, psi_eta_u = char_funcs(W, eta, tau_u)
    root = np.sqrt(np.abs(psi_eta_u))
    # psi_zeta is compactly supported, so the ratio is never formed
    # outside |nu tau| <= 1, where the root stays bounded away from 0
    weight = smoothing_cf(nu_U * tau_u)
    integ_u = np.where(weight > 0,
                       psi_W_u * weight / np.maximum(root, 1e-300), 0.0)
    f_U = np.array([
        float(np.real(np.trapezoid(integ_u * np.exp(-1j * tau_u * u), tau_u))
              / (2 * np.pi)) for u in np.atleast_1d(grid_u)])
    return f_U, f_eps


def cheatsheet():
    return "_hrz_paneldec: psi_eps = psi_eta^{1/2} needs eps SYMMETRIC; else the root's sign is unidentified"
