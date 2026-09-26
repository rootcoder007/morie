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

__all__ = ["panel_residuals", "char_funcs", "smoothing_cf", "deconvolve_pair",
           "default_bandwidths", "KERNELS"]

KERNELS = ("fourfold", "flattop")
# f_U flat-top cut-off: the first tau where |psi_nW| drops to this many
# sampling standard errors (1/sqrt(N)); past it the ratio is noise
_FLATTOP_FLOOR = 2.0


def _check_kernel(kernel):
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}.")
    return kernel


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


def smoothing_cf(u, kernel="fourfold"):
    r"""``psi_zeta``: a bounded real characteristic function supported
    on [-1, 1].

    The book's example (Horowitz, Sec. 5.1.3 and assumption PHU7) is
    the fourfold convolution of the U[-1/4, 1/4] density with itself:
    the density of a sum of four such uniforms, a cubic B-spline on
    [-1, 1].  Its Fourier transform is sinc^4 >= 0, so after dividing by
    its value 4/3 at the origin it is a genuine characteristic function
    with psi_zeta(0) = 1 and two continuous derivatives.  With
    x = 2u + 2 the Irwin-Hall n = 4 density gives
    psi_zeta(u) = (3/2) * (1/6) * sum_k (-1)^k C(4, k) (x - k)_+^3.
    Compact support in tau is the whole point: it is what stops the
    integrand being evaluated where the denominator vanishes.

    ``kernel="flattop"`` is instead the indicator of [-1, 1], the sinc
    (flat-top) deconvolution kernel: its bias is of infinite order for
    smooth densities, so it is markedly more accurate, but it is not a
    characteristic function and so falls outside assumption PHU7.
    """
    _check_kernel(kernel)
    if kernel == "flattop":
        uv = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
        return np.asarray([1.0 if abs(v) <= 1.0 else 0.0 for v in uv.tolist()],
                          dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    out = []
    for v in u.tolist():
        if abs(v) >= 1.0:
            out.append(0.0)
            continue
        x = 2.0 * abs(v) + 2.0
        ih = (x ** 3 - 4.0 * (x - 1.0) ** 3 + 6.0 * (x - 2.0) ** 3
              - 4.0 * max(x - 3.0, 0.0) ** 3) / 6.0
        out.append(1.5 * ih)
    return np.asarray(out, dtype=float)


def default_bandwidths(eta, n, kernel="fourfold", W=None):
    r"""Default (nu_U, nu_eps) from the scale of the error.

    ``sigma_eps`` is estimated as sd(eta) / sqrt(2), eta being the
    difference of two independent errors.  f_nU divides by
    |psi_n eta|^{1/2} ~ psi_eps, so its cut-off 1/nu_U may only grow as
    fast as the noise amplification exp(sigma^2 / (2 nu^2)) / n stays
    bounded: nu_U = sigma_eps / sqrt(log n), the same criterion as the
    Section 5.1 estimator.  f_n eps involves no division, so it is an
    ordinary smoothing problem at the Silverman rate:
    nu_eps = 0.5 sigma_eps N^{-1/5} with N the number of differences.
    The psi_zeta kernel has variance 12 nu^2, which is why both are much
    smaller than a (log n)^{-1/2} default.  Checked against the true
    densities in simulations (U ~ N(0, 1), eps ~ N(0, s^2),
    s = 0.25 .. 1, n = 400 .. 3000): sup errors 2-7 times smaller than
    the old (log n)^{-1/2} default.
    """
    e = [float(v) for v in np.asarray(eta, dtype=float).ravel().tolist()]
    N = len(e)
    if N < 2:
        raise ValueError("need at least two differences to set a bandwidth")
    m = sum(e) / N
    sig = (sum((v - m) ** 2 for v in e) / (N - 1)) ** 0.5 / 2.0 ** 0.5
    if not sig > 0:
        raise ValueError("the differenced errors have zero spread")
    import math
    if _check_kernel(kernel) == "fourfold":
        return sig / math.sqrt(math.log(n)), 0.5 * sig * N ** -0.2
    # flat-top: f_eps (no division) cuts off where the noise amplification
    # of the square root stays bounded, nu = sigma_eps / sqrt(log n);
    # f_U cuts off where |psi_nW| reaches its own sampling noise floor --
    # beyond it the ratio carries no signal (the book's "psi_nU(1/nu)
    # approximately zero" rule applied to the estimable numerator)
    if W is None:
        raise ValueError("the flat-top f_U bandwidth needs the W residuals")
    w = [float(v) for v in np.asarray(W, dtype=float).ravel().tolist()]
    NW = len(w)
    mw = sum(w) / NW
    sw = (sum((v - mw) ** 2 for v in w) / (NW - 1)) ** 0.5
    floor = _FLATTOP_FLOOR / math.sqrt(NW)
    step = 0.02 / sw
    T = 1500 * step
    for k_ in range(1, 1501):
        t = k_ * step
        re_ = sum(math.cos(t * v) for v in w) / NW
        im_ = sum(math.sin(t * v) for v in w) / NW
        if math.hypot(re_, im_) < floor:
            T = t
            break
    return 1.0 / T, sig / math.sqrt(math.log(n))


def deconvolve_pair(W, eta, grid_u, grid_z, nu_U, nu_eps, n_tau=2001,
                    kernel="fourfold"):
    """(5.25) and (5.26): smoothed estimators of f_eps and f_U."""
    nu_U = float(nu_U)
    nu_eps = float(nu_eps)
    if nu_U <= 0 or nu_eps <= 0:
        raise ValueError(f"bandwidths must be positive, got {(nu_U, nu_eps)}.")
    tau_u = np.linspace(-1.0 / nu_U, 1.0 / nu_U, int(n_tau))
    tau_e = np.linspace(-1.0 / nu_eps, 1.0 / nu_eps, int(n_tau))

    _, psi_eta_e = char_funcs(W, eta, tau_e)
    # (5.25)
    integ_e = np.sqrt(np.abs(psi_eta_e)) * smoothing_cf(nu_eps * tau_e, kernel)
    f_eps = np.array([
        float(np.real(np.trapezoid(integ_e * np.exp(-1j * tau_e * z), tau_e))
              / (2 * np.pi)) for z in np.atleast_1d(grid_z)])

    psi_W_u, psi_eta_u = char_funcs(W, eta, tau_u)
    root = np.sqrt(np.abs(psi_eta_u))
    # psi_zeta is compactly supported, so the ratio is never formed
    # outside |nu tau| <= 1, where the root stays bounded away from 0
    weight = smoothing_cf(nu_U * tau_u, kernel)
    integ_u = np.where(weight > 0,
                       psi_W_u * weight / np.maximum(root, 1e-300), 0.0)
    f_U = np.array([
        float(np.real(np.trapezoid(integ_u * np.exp(-1j * tau_u * u), tau_u))
              / (2 * np.pi)) for u in np.atleast_1d(grid_u)])
    return f_U, f_eps


def cheatsheet():
    return "_hrz_paneldec: psi_eps = psi_eta^{1/2} needs eps SYMMETRIC; else the root's sign is unidentified"
