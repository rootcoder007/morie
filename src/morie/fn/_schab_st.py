# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatio-temporal covariance, semivariogram and point-process primitives.

Schabenberger & Gotway (2005), Chapter 9. One implementation per book
equation, so the four spatio-temporal front-end modules stay thin and
cannot drift apart.

The chapter's central warning drives the design. Sec. 9.1 is explicit that
treating spatio-temporal data as a field in R^{d+1} "is not encouraged":
eq (9.2) shows the naive isotropic exponential in R^3 is a *valid*
correlation function that makes no practical sense, because the spatial
range, the temporal range and the units are not comparable. So nothing here
ever concatenates t onto the coordinate vector. Space lag ||h|| and time lag
|k| are carried separately through every function, and any model that mixes
them does so through named parameters (the anisotropy pair of eq (9.3), or
Gneiting's ψ).

Validity is the other theme. Eq (9.5) states positive definiteness for the
spatio-temporal case, and Sec. 9.3 notes that Gneiting (2002) found some of
the covariance functions published in Cressie and Huang (1999) to be
invalid. `is_valid_covariance` therefore checks (9.5) numerically on the
actual design rather than trusting the construction.

Everything here is internal.
"""

import numpy as np

__all__ = []

# numpy renamed trapz -> trapezoid in 2.0; bind once so the choice is not
# re-decided inside a quadrature loop.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# lag handling
# --------------------------------------------------------------------------

def _as_lags(h, k):
    """Broadcast a spatial lag and a temporal lag to a common shape.

    Kept separate deliberately -- see the module docstring on eq (9.2).
    """
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    if np.any(h < 0):
        raise ValueError("spatial lag `h` must be non-negative")
    h, k = np.broadcast_arrays(h, np.abs(k))
    return h, k


def st_lag_matrices(coords, times):
    """Pairwise spatial distances and absolute time lags.

    Returns (D, K) with D[i, j] = ||s_i - s_j|| and K[i, j] = |t_i - t_j|,
    the "two distances between points in a space-time process" of Sec. 9.1.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    times = np.asarray(times, dtype=float).ravel()
    if coords.shape[0] != times.size:
        raise ValueError("`coords` and `times` must have the same length")
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    k = np.abs(times[:, None] - times[None, :])
    return d, k


# --------------------------------------------------------------------------
# Sec. 9.2 -- separable covariance functions
# --------------------------------------------------------------------------

def separable_covariance(h, k, cov_spatial, cov_temporal, form="product"):
    """The separable forms of Sec. 9.2.

    ``product``      C(h,k) = Cs(h) Ct(k)
    ``sum``          C(h,k) = Cs(h) + Ct(k)
    ``product_sum``  C(h,k) = Cs(h) Ct(k) + Cs(h) + Ct(k)

    The first two are the additive and multiplicative constructions the text
    gives; both are valid whenever the components are, by the two elementary
    properties quoted at the head of Sec. 9.2 (a non-negative combination and
    a product of valid covariance functions are valid).

    ``product_sum`` is De Cesare, Myers and Posa (2001). It is included here
    and NOT under the non-separable heading only because the book introduces
    it in this section; the text is explicit that it "is generally
    nonseparable", so `is_separable` returns False for it.
    """
    h, k = _as_lags(h, k)
    cs = np.asarray(cov_spatial(h), dtype=float)
    ct = np.asarray(cov_temporal(k), dtype=float)
    if form == "product":
        return cs * ct
    if form == "sum":
        return cs + ct
    if form == "product_sum":
        return cs * ct + cs + ct
    raise ValueError("`form` must be 'product', 'sum' or 'product_sum'")


def is_separable(form):
    """Whether a `separable_covariance` form is genuinely separable."""
    if form in ("product", "sum"):
        return True
    if form == "product_sum":
        return False                      # De Cesare et al.; see Sec. 9.2
    raise ValueError(f"unknown form {form!r}")


def anisotropic_correlation(h, k, theta_s, theta_t, corr_fn):
    """eq (9.3): Corr = R(theta_s ||h||^2 + theta_t k^2).

    R must be a valid correlation function in R^{d+1}. theta_s and theta_t
    are the spatial and temporal anisotropy parameters -- the text's point is
    that without them the "added dimension" construction (9.2) implicitly
    asserts that a unit of time equals a unit of distance.
    """
    h, k = _as_lags(h, k)
    if theta_s <= 0 or theta_t <= 0:
        raise ValueError("anisotropy parameters must be positive")
    return np.asarray(corr_fn(theta_s * h**2 + theta_t * k**2), dtype=float)


def exponential_separable_correlation(h, k, theta_s, theta_t):
    """eq (9.4), the worked separable example.

    Corr = exp{-theta_s ||h||} exp{-theta_t |k|}. With evenly spaced time
    points the temporal factor is an AR(1) structure and the spatial factor
    is exponential -- the Mitchell and Gumpertz (2003) model.
    """
    h, k = _as_lags(h, k)
    if theta_s <= 0 or theta_t <= 0:
        raise ValueError("`theta_s` and `theta_t` must be positive")
    return np.exp(-theta_s * h) * np.exp(-theta_t * k)


def posa_covariance(h, k_times, cov_spatial, sill_fn):
    """eq (9.6): Cov = sigma^2(t) Cs(h), Posa (1993).

    Only the sill is time dependent, so the process is non-stationary in
    time. The text's observation is worth preserving: kriging predictions in
    a no-nugget model are invariant to scaling of the covariance function, so
    the predictions at a given time are unaffected by this non-stationarity
    -- but the kriging VARIANCE is not invariant. Returns both the covariance
    and that warning as a flag.
    """
    h = np.asarray(h, dtype=float)
    t = np.asarray(k_times, dtype=float)
    cs = np.asarray(cov_spatial(h), dtype=float)
    s2 = np.asarray(sill_fn(t), dtype=float)
    return s2 * cs


# --------------------------------------------------------------------------
# Sec. 9.3.1 -- Gneiting's monotone function approach
# --------------------------------------------------------------------------

def gneiting_covariance(h, k, sigma2=1.0, a=1.0, c=1.0, alpha=1.0, beta=1.0,
                        gamma=1.0, d=2):
    """eq (9.8), Gneiting (2002).

                     sigma^2                 c ||h||^{2 gamma}
      C(h,k) = --------------------- exp{ - --------------------- }
                (a|k|^{2 alpha}+1)^beta      (a|k|^{2 alpha}+1)^{beta gamma}

    This is (9.7) with phi(t) = exp{-c t^gamma} and psi(t) = (a t^alpha + 1)^beta,
    which the text states satisfy the requirements (phi completely monotone,
    psi positive with a completely monotone derivative) for

        c, a > 0;   0 < gamma, alpha <= 1;   0 <= beta <= 1.

    Those bounds are enforced, not assumed: outside them the construction
    carries no validity guarantee and the function would silently return a
    non-positive-definite model.

    Checked against Gneiting's own class, not only the book's transcription.
    The general form is C(r,u) = sigma^2 psi(u^2)^{-d/2} g(r^2 / psi(u^2)) with
    g completely monotone and psi strictly positive with a completely monotone
    derivative, normalised so g(0) = psi(0) = 1 and hence C(0,0) = sigma^2 --
    which this returns. Note the exponents are on the OUTER psi: with the inner
    bracket b = a|k|^{2 alpha} + 1 and psi = b^beta, psi^{d/2} = b^{beta d / 2}
    and psi^{gamma} = b^{beta gamma}, which is what the code computes.

    One correction to how these bounds are usually described: Gneiting (2002)
    gives SUFFICIENT conditions for validity, not necessary ones. Zastavnyi and
    Porcu (2011) later established the necessary conditions and relaxed the
    hypothesis on psi. Enforcing Gneiting's bounds is therefore conservative --
    it may reject some parameter sets that are in fact valid. That is the right
    default for a library, but it is a choice, not a law.

    The general (9.7) form carries psi^{d/2} in the denominator. The book
    writes (9.8) out for d = 2, where beta and d/2 = 1 coincide; `d` is kept
    as an argument so the general case is available and the d = 2 default
    reproduces the printed equation exactly.
    """
    h, k = _as_lags(h, k)
    if a <= 0 or c <= 0:
        raise ValueError("`a` and `c` must be positive")
    if not (0 < gamma <= 1):
        raise ValueError("`gamma` must lie in (0, 1]")
    if not (0 < alpha <= 1):
        raise ValueError("`alpha` must lie in (0, 1]")
    if not (0 <= beta <= 1):
        raise ValueError("`beta` must lie in [0, 1]")
    if sigma2 < 0:
        raise ValueError("`sigma2` must be non-negative")
    psi = a * k ** (2.0 * alpha) + 1.0
    return (sigma2 / psi ** (beta * d / 2.0)) * np.exp(
        -c * h ** (2.0 * gamma) / psi ** (beta * gamma))


def gneiting_with_temporal(h, k, sigma2=1.0, a=1.0, c=1.0, alpha=1.0,
                           beta=1.0, beta_t=1.0, gamma=1.0, d=2):
    """eq (9.9): Ct(k) x C(h,k), still a valid spatio-temporal covariance.

    At ||h|| = 0 eq (9.8) reduces to the purely temporal covariance
    (a|k|^{2 alpha}+1)^{-beta}, so multiplying by another such factor simply
    adds the exponents -- which is why (9.9) carries beta_t + beta.

    The point of this form is that it is separable at beta = 0 and
    non-separable otherwise, and the two are NESTED. Sec. 9.3.1 spells out
    the consequence: fit by (restricted) maximum likelihood with and without
    beta = 0 and compare twice the negative log likelihood to test
    H0: beta = 0. Because the null sits on the boundary of the parameter
    space the naive chi-square reference is wrong; the text points to Self
    and Liang (1987) for the correction. `separability_test` below applies it.
    """
    h, k = _as_lags(h, k)
    if beta_t < 0:
        raise ValueError("`beta_t` must be non-negative")
    if a <= 0 or c <= 0:
        raise ValueError("`a` and `c` must be positive")
    if not (0 < gamma <= 1) or not (0 < alpha <= 1) or not (0 <= beta <= 1):
        raise ValueError("Gneiting parameter bounds violated")
    psi = a * k ** (2.0 * alpha) + 1.0
    return (sigma2 / psi ** (beta_t + beta * d / 2.0)) * np.exp(
        -c * h ** (2.0 * gamma) / psi ** (beta * gamma))


def separability_test(neg2_loglik_unrestricted, neg2_loglik_separable):
    """H0: beta = 0 in eq (9.9), with the boundary correction.

    The statistic is the usual difference in twice the negative (restricted)
    log likelihood. Under H0 the parameter sits on the boundary of the
    parameter space, and Sec. 6.2.3 states the consequence and the remedy
    directly: the statistic is then "a mixture of a degenerate distribution
    giving probability 1 to the value zero, and a chi^2 distribution with
    dim(theta_2) - dim(theta_1) degrees of freedom (Self and Liang, 1987;
    Littell, Milliken, Stroup and Wolfinger, 1996). Thus, to make the test,
    simply divide the p-value obtained from a chi^2 with
    dim(theta_2) - dim(theta_1) degrees of freedom by 2."

    So the halving is the book's own prescription, not an inference from the
    name Self and Liang. Reporting the naive p-value would reject separability
    too rarely -- conservative, but wrong in the direction that matters when
    the whole point is to justify the simpler model.
    """
    stat = float(neg2_loglik_separable) - float(neg2_loglik_unrestricted)
    if stat < 0:
        stat = 0.0
    p_naive = _chi2_sf_1df(stat)
    return {"statistic": stat,
            "p_value": 0.5 * p_naive,
            "p_value_naive_chi2_1": p_naive,
            "reference": "0.5 chi^2_0 + 0.5 chi^2_1 (Self and Liang, 1987)"}


def _chi2_sf_1df(x):
    """P(chi^2_1 > x) = erfc(sqrt(x/2)); native, no scipy."""
    from math import erfc, sqrt
    x = float(x)
    if x <= 0:
        return 1.0
    return float(erfc(sqrt(x / 2.0)))


# --------------------------------------------------------------------------
# Sec. 9.3.3 -- mixture approaches
# --------------------------------------------------------------------------

def power_mixture_correlation(rs, rt, distribution="poisson", **params):
    """eqs (9.14) and Example 9.1, Ma (2002).

    The univariate positive power mixture is

        R(h,k) = sum_i (Rs(h) Rt(k))^i pi_i,

    which the text observes IS the probability generating function of the
    mixing distribution evaluated at w = Rs(h) Rt(k). So the construction
    reduces to: take a pgf, substitute the correlation product for w. The
    series is never summed here -- the closed-form pgf is used, which is both
    exact and what the book actually prescribes.

    Example 9.1 gives the two cases implemented:
      Binomial(n, pi):  G(w) = (pi(w-1)+1)^n
      Poisson(lambda):  G(w) = exp{lambda(w-1)}

    Validity rests on R(u)^i being a valid correlation model in R^d for any
    positive integer i whenever R(u) is -- so the mixture is a non-negative
    combination of valid models.
    """
    rs = np.asarray(rs, dtype=float)
    rt = np.asarray(rt, dtype=float)
    if np.any(np.abs(rs) > 1 + 1e-12) or np.any(np.abs(rt) > 1 + 1e-12):
        raise ValueError("`rs` and `rt` must be correlations in [-1, 1]")
    w = rs * rt
    if distribution == "poisson":
        lam = float(params.get("lam", 1.0))
        if lam <= 0:
            raise ValueError("`lam` must be positive")
        return np.exp(lam * (w - 1.0))
    if distribution == "binomial":
        n = int(params.get("n", 1))
        pi = float(params.get("pi", 0.5))
        if n < 1 or not (0.0 <= pi <= 1.0):
            raise ValueError("`n` >= 1 and `pi` in [0, 1] required")
        return (pi * (w - 1.0) + 1.0) ** n
    raise ValueError("`distribution` must be 'poisson' or 'binomial'")


def bivariate_power_mixture_correlation(rs, rt, pmf):
    """eq (9.13), the bivariate positive power mixture of Ma (2002).

        R(h,k) = sum_i sum_j Rs(h)^i Rt(k)^j pi_ij

    `pmf` is the (I x J) mass matrix of [U, V] on the non-negative integers.
    The text notes bivariate discrete mass functions "are quite rare", which
    is why the univariate form above is the usual route; this is kept for
    completeness and to make the univariate case checkable against it.
    """
    rs = np.asarray(rs, dtype=float)
    rt = np.asarray(rt, dtype=float)
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    if np.any(pmf < 0):
        raise ValueError("`pmf` must be non-negative")
    total = float(pmf.sum())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"`pmf` must sum to 1 (got {total:.12g})")
    i = np.arange(pmf.shape[0])
    j = np.arange(pmf.shape[1])
    powers_s = rs[..., None] ** i                      # ... x I
    powers_t = rt[..., None] ** j                      # ... x J
    return np.einsum("...i,...j,ij->...", powers_s, powers_t, pmf)


def scale_mixture_covariance(h, k, cov_spatial, cov_temporal, nodes, weights):
    """eq (9.16), the univariate scale mixture of Ma (2002).

        C(h,k) = integral Cs(h u) Ct(k u) dF(u)

    The construction is Z(s,t) = Zs(sU) Zt(tV): the spatial and temporal
    coordinates are made to depend on a common random scale, which is what
    introduces space-time interaction into what would otherwise be a product
    covariance. `nodes` and `weights` discretise F; weights must be
    non-negative and sum to one, since F is a distribution function.

    De Iaco, Myers and Posa (2002) generalise F to a positive measure and
    apply the same mixing to product-sum functions; that extension is not
    implemented, and the weight normalisation here is what keeps this
    function honest about being the (9.16) special case.
    """
    h, k = _as_lags(h, k)
    u = np.asarray(nodes, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if u.size != w.size:
        raise ValueError("`nodes` and `weights` must have the same length")
    if np.any(w < 0):
        raise ValueError("`weights` must be non-negative (F is a d.f.)")
    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError(f"`weights` must sum to 1 (got {w.sum():.12g})")
    if np.any(u < 0):
        raise ValueError("scale `nodes` must be non-negative")
    out = np.zeros(np.broadcast(h, k).shape, dtype=float)
    for ui, wi in zip(u, w):
        out += wi * (np.asarray(cov_spatial(h * ui), dtype=float)
                     * np.asarray(cov_temporal(k * ui), dtype=float))
    return out


def bivariate_scale_mixture_covariance(h, k, cov_spatial, cov_temporal,
                                       nodes_u, nodes_v, weights):
    """eq (9.15): C(h,k) = integral Cs(h u) Ct(k v) dF(u,v)."""
    h, k = _as_lags(h, k)
    u = np.asarray(nodes_u, dtype=float).ravel()
    v = np.asarray(nodes_v, dtype=float).ravel()
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    if w.shape != (u.size, v.size):
        raise ValueError("`weights` must be shaped (len(nodes_u), len(nodes_v))")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError("`weights` must be non-negative and sum to 1")
    out = np.zeros(np.broadcast(h, k).shape, dtype=float)
    for a, ui in enumerate(u):
        cs = np.asarray(cov_spatial(h * ui), dtype=float)
        for b, vi in enumerate(v):
            if w[a, b] == 0.0:
                continue
            out += w[a, b] * cs * np.asarray(cov_temporal(k * vi), dtype=float)
    return out


# --------------------------------------------------------------------------
# Sec. 9.3.4 -- the differential equation approach
# --------------------------------------------------------------------------

def gauss_legendre(n):
    """Nodes and weights on [-1, 1] by Golub-Welsch.

    Golub and Welsch (1969), Mathematics of Computation 23(106):221-230 --
    NOT a Schabenberger & Gotway result; that book does not cite it. Given the
    three-term recurrence, the nodes are the eigenvalues of the symmetric
    tridiagonal Jacobi matrix and the weights are

        w_i = mu_0 * (first component of the normalised eigenvector)^2,

    where mu_0 is the ZEROTH MOMENT of the weight function. For Legendre on
    [-1, 1] the recurrence gives zero diagonal and off-diagonal
    k / sqrt(4k^2 - 1), and mu_0 = integral_{-1}^{1} dx = 2, hence the factor
    of 2 below. The same algorithm in `_schab_hermite.gauss_hermite` carries
    NO such factor, because there the weight is the standard Gaussian density
    and mu_0 = 1. Getting mu_0 wrong rescales every quadrature silently.

    Written out rather than taken from a library so the R arm runs the same
    arithmetic -- both languages then need only a symmetric eigensolver.
    """
    n = int(n)
    if n < 1:
        raise ValueError("`n` must be positive")
    k = np.arange(1.0, n)
    off = k / np.sqrt(4.0 * k * k - 1.0)
    jac = np.diag(off, 1) + np.diag(off, -1)
    vals, vecs = np.linalg.eigh(jac)
    return vals, 2.0 * vecs[0, :] ** 2


def bessel_j0(x, n_quad=200):
    """J_0(x) from its integral representation, native.

        J_0(x) = (1/pi) integral_0^pi cos(x sin theta) dtheta

    The integrand is smooth and periodic, so the trapezoid rule converges
    geometrically -- far better here than a rational approximation, and it is
    trivially identical in R. Used only by `jones_zhang_covariance`, where
    (9.17) is a zero-order Hankel transform.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    theta = np.linspace(0.0, np.pi, int(n_quad) + 1)
    integrand = np.cos(x[..., None] * np.sin(theta))
    return _TRAPZ(integrand, theta, axis=-1) / np.pi


def whittle_spatial_covariance(h, sigma2=1.0, theta=1.0):
    """Cs(||h||) = sigma_s^2 theta ||h|| K_1(theta ||h||), Sec. 9.3.4.

    Whittle's (1954) elementary process, the solution of the stochastic
    Laplace equation. K_1 is computed from its own integral representation
    K_1(z) = integral_0^inf exp{-z cosh u} cosh u du, which keeps the module
    free of a special-function dependency. The limit at h = 0 is sigma^2:
    z K_1(z) -> 1 as z -> 0.
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    if np.any(h < 0):
        raise ValueError("lag `h` must be non-negative")
    if theta <= 0:
        raise ValueError("`theta` must be positive")
    z = theta * h
    out = np.full(z.shape, float(sigma2))
    pos = z > 0
    if np.any(pos):
        out[pos] = sigma2 * z[pos] * _bessel_k1(z[pos])
    return out


def _bessel_k1(z, upper=40.0, n_quad=400):
    """K_1(z) = integral_0^inf exp{-z cosh u} cosh u du, by Gauss-Legendre."""
    z = np.asarray(z, dtype=float)
    x, w = gauss_legendre(int(n_quad))
    u = 0.5 * upper * (x + 1.0)
    wu = 0.5 * upper * w
    ch = np.cosh(u)
    return np.einsum("j,...j->...", wu, np.exp(-z[..., None] * ch) * ch)


def jones_zhang_covariance(h, k, sigma2=1.0, theta=1.0, c=1.0, p=1.5, d=2,
                           n_quad=400, upper=None):
    """eq (9.17), Jones and Zhang (1997), for d = 2.

                   sigma^2   inf   tau exp{-(k/c)(tau^2+theta^2)^p}
        C(h,k) =  --------- integral --------------------------------- J_0(tau h) dtau
                    4 c pi    0            (tau^2 + theta^2)^p

    The text notes this is a zero-order Hankel transform, exactly as the
    purely spatial isotropic case of eqs (4.7)-(4.8), and that p governs the
    smoothness of the process and must exceed max{1, d/2}.

    That constraint is not decoration. At k = 0 the integrand behaves like
    tau^{1 - 2p} for large tau, so the integral converges only for p > 1 --
    precisely the stated bound with d = 2. It is enforced.

    The infinite upper limit is truncated. Rather than pick a cutoff and hope,
    the truncation point is chosen from the decay and the discarded tail is
    BOUNDED and returned, so a caller can see when the quadrature is not to be
    trusted instead of receiving a confidently wrong number.
    """
    h, k = _as_lags(h, k)
    if theta <= 0 or c <= 0 or sigma2 < 0:
        raise ValueError("`theta` and `c` must be positive, `sigma2` >= 0")
    if p <= max(1.0, d / 2.0):
        raise ValueError(
            f"`p` must exceed max(1, d/2) = {max(1.0, d / 2.0)} for the "
            f"integral in eq (9.17) to converge (got p = {p})")

    hflat = np.atleast_1d(np.asarray(h, dtype=float)).ravel()
    kflat = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
    out = np.empty(hflat.size, dtype=float)
    reached, rels, bounds = [], [], []
    for idx in range(hflat.size):
        val, t_end, rel, bnd = _hankel_j0_panels(
            hflat[idx], kflat[idx], theta, c, p, n_quad=n_quad,
            max_upper=upper)
        out[idx] = val
        reached.append(t_end)
        rels.append(rel)
        bounds.append(bnd)
    out *= sigma2 / (4.0 * c * np.pi)
    scale = sigma2 / (4.0 * c * np.pi)
    return out.reshape(np.broadcast(h, k).shape), {
        "upper_reached": float(np.max(reached)),
        "n_quad": int(n_quad),
        "last_panel_rel": float(np.max(rels)),
        "tail_bound": float(np.max(bounds) * scale)}


def _tail_bound_j0(t, h, p):
    """Analytic bound on the discarded tail of eq (9.17).

    Beyond T the integrand is at most tau^{1-2p} in magnitude (the exponential
    factor is <= 1 and (tau^2+theta^2)^p >= tau^{2p}). For h > 0 the standard
    bound |J_0(z)| <= sqrt(2/(pi z)) buys another half power, which matters:
    without it the bound is so pessimistic that no achievable truncation ever
    looks acceptable.

    Neither bound is from Schabenberger & Gotway; both are standard properties
    of J_0. The book supplies eq (9.17) and the constraint p > max{1, d/2}, and
    nothing about how to evaluate the transform numerically -- so the quadrature
    strategy here is engineering around a book formula, not a book method, and
    is reported with its own diagnostics for exactly that reason.
    """
    if t <= 0:
        return np.inf
    if h > 0 and p > 1.25:
        return float(np.sqrt(2.0 / (np.pi * h)) * t ** (2.5 - 2.0 * p)
                     / (2.0 * p - 2.5))
    if p > 1.0:
        return float(t ** (2.0 - 2.0 * p) / (2.0 * p - 2.0))
    return np.inf


def _hankel_j0_panels(hval, kval, theta, c, p, n_quad=40, rtol=1e-10,
                      max_upper=None, max_panels=20000, quiet_runs=4):
    """Integrate eq (9.17) panel by panel outward from zero.

    Two things force this rather than one Gauss-Legendre rule on [0, T].
    J_0(tau h) oscillates with period 2 pi / h, so a single rule over a long
    interval aliases it; panels are sized to that period. And the integrand
    decays only algebraically at k = 0, so T cannot be fixed in advance.

    Termination is on the ACHIEVED panel contribution -- `quiet_runs`
    consecutive panels each below rtol of the running total -- rather than on
    the analytic bound, which is far too pessimistic to ever be met. The bound
    is still computed and returned, so a caller can see how much of the tail
    is merely argued away rather than integrated.
    """
    x, w = gauss_legendre(int(n_quad))
    if hval > 0:
        panel = min(np.pi / hval, max(1.0, 2.0 * theta))
    else:
        panel = max(1.0, 2.0 * theta)
    panel = max(panel, 1e-3)
    if max_upper is None:
        max_upper = np.inf

    acc = 0.0
    t = 0.0
    quiet = 0
    panels = 0
    last_rel = np.inf
    while panels < max_panels and t < max_upper:
        a0, b0 = t, min(t + panel, max_upper)
        tau = 0.5 * (b0 - a0) * (x + 1.0) + a0
        wt = 0.5 * (b0 - a0) * w
        q = tau**2 + theta**2
        core = tau * np.exp(-(kval / c) * q**p) / q**p
        contrib = float(np.sum(wt * core * bessel_j0(tau * hval)))
        acc += contrib
        t = b0
        panels += 1
        denom = max(abs(acc), 1e-300)
        last_rel = abs(contrib) / denom
        quiet = quiet + 1 if last_rel < rtol else 0
        if quiet >= quiet_runs:
            break
    return acc, t, last_rel, _tail_bound_j0(t, hval, p)


def jones_zhang_separable(h, k, sigma2=1.0, theta_s=1.0, theta_t=1.0):
    """The separable case of Sec. 9.3.4.

        C(h,k) = Ct(k) Cs(h) = sigma^2 exp{-theta_t k} theta_s ||h|| K_1(theta_s ||h||)

    This is what the combined stochastic equation yields when the spatial
    Laplace operator and the temporal derivative act independently -- a
    product separable model with a Whittle spatial component.
    """
    h, k = _as_lags(h, k)
    return (np.exp(-theta_t * k)
            * whittle_spatial_covariance(h.ravel(), sigma2, theta_s
                                         ).reshape(h.shape))


# --------------------------------------------------------------------------
# eq (9.5) -- validity
# --------------------------------------------------------------------------

def st_covariance_matrix(coords, times, cov_fn):
    """The (n x n) covariance matrix implied by cov_fn on the design."""
    d, k = st_lag_matrices(coords, times)
    return np.asarray(cov_fn(d, k), dtype=float)


def is_valid_covariance(coords, times, cov_fn, tol=None):
    """Check eq (9.5) numerically on the given design.

        sum_i sum_j a_i a_j C(s_i - s_j, t_i - t_j) >= 0

    for all real a, which is positive semi-definiteness of the matrix above.
    Checked by symmetric eigendecomposition rather than by sampling random a:
    the minimum eigenvalue IS the minimum of the quadratic form over unit a,
    so a single eigensolve answers the question that no finite number of
    random draws can.

    Sec. 9.3 records that Gneiting (2002) found covariance functions
    published in Cressie and Huang (1999) to be invalid -- construction alone
    is not proof, which is why this exists.
    """
    sigma = st_covariance_matrix(coords, times, cov_fn)
    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=0.0):
        return {"valid": False, "min_eigenvalue": np.nan,
                "reason": "covariance matrix is not symmetric"}
    vals = np.linalg.eigvalsh(sigma)
    lo = float(vals.min())
    scale = float(np.abs(vals).max())
    if tol is None:
        tol = -1e-10 * max(scale, 1.0)
    return {"valid": bool(lo >= tol), "min_eigenvalue": lo,
            "max_eigenvalue": float(vals.max()), "tolerance": float(tol),
            "reason": "" if lo >= tol else "minimum eigenvalue is negative"}


# --------------------------------------------------------------------------
# Sec. 9.4 -- the spatio-temporal semivariogram
# --------------------------------------------------------------------------

def semivariogram_from_covariance(h, k, cov_fn):
    """gamma(h,k) = C(0,0) - C(h,k), Sec. 9.4.

    The identity holds for a stationary process, exactly as in the purely
    spatial case: gamma = Var[Z] - Cov, and Var[Z] = C(0,0).
    """
    h, k = _as_lags(h, k)
    c0 = float(np.asarray(cov_fn(np.array(0.0), np.array(0.0))).ravel()[0])
    return c0 - np.asarray(cov_fn(h, k), dtype=float)


def empirical_st_semivariogram(coords, times, z, n_space_bins=10,
                               n_time_bins=5, max_dist=None, max_time=None):
    """eq (9.18), the spatio-temporal Matheron estimator.

        gamma_hat(h,k) = 1 / (2 |N(h,k)|) sum_{N(h,k)} (Z(s_i,t_i) - Z(s_j,t_j))^2

    N(h,k) is the set of pairs within spatial distance h AND time lag k of
    each other. Because the data are generally irregular in both space and
    time, pairs are collected into lag CLASSES; the text is explicit that the
    spatial and temporal tolerances must be chosen separately, "to accommodate
    a sufficient number of point pairs at each spatio-temporal lag". Two
    independent bin counts is what implements that.

    Returns the (n_space_bins x n_time_bins) grids of gamma_hat, the pair
    counts |N(h,k)|, and the bin centres. Cells with no pairs are NaN and
    their count is 0 -- not silently zero-filled, since a zero semivariogram
    and an unestimated one are different claims.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    times = np.asarray(times, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    if coords.shape[0] != n or times.size != n:
        raise ValueError("`coords`, `times` and `z` must have the same length")
    if n < 2:
        raise ValueError("need at least two observations")

    i, j = np.triu_indices(n, k=1)
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    u = np.abs(times[i] - times[j])
    sq = (z[i] - z[j]) ** 2

    if max_dist is None:
        max_dist = float(d.max()) / 2.0 if d.size else 1.0
    if max_time is None:
        max_time = float(u.max()) / 2.0 if u.size else 1.0
    if max_dist <= 0 or max_time <= 0:
        raise ValueError("`max_dist` and `max_time` must be positive")

    keep = (d <= max_dist) & (u <= max_time)
    d, u, sq = d[keep], u[keep], sq[keep]

    d_edges = np.linspace(0.0, max_dist, int(n_space_bins) + 1)
    u_edges = np.linspace(0.0, max_time, int(n_time_bins) + 1)
    di = np.clip(np.digitize(d, d_edges) - 1, 0, int(n_space_bins) - 1)
    ui = np.clip(np.digitize(u, u_edges) - 1, 0, int(n_time_bins) - 1)

    counts = np.zeros((int(n_space_bins), int(n_time_bins)), dtype=int)
    total = np.zeros_like(counts, dtype=float)
    np.add.at(counts, (di, ui), 1)
    np.add.at(total, (di, ui), sq)

    gamma = np.full(counts.shape, np.nan)
    nz = counts > 0
    gamma[nz] = total[nz] / (2.0 * counts[nz])
    return {"gamma": gamma, "counts": counts,
            "space_lags": 0.5 * (d_edges[:-1] + d_edges[1:]),
            "time_lags": 0.5 * (u_edges[:-1] + u_edges[1:]),
            "space_edges": d_edges, "time_edges": u_edges}


def conditional_spatial_semivariogram(coords, times, z, at_time,
                                      n_bins=10, max_dist=None, tol=0.0):
    """eq (9.19), the conditional spatial semivariogram at a single time.

        gamma_hat_t(h) = 1 / (2 |N_t(h)|) sum_{N_t(h)} (Z(s_i,t) - Z(s_j,t))^2

    The text is careful to distinguish this from (9.18): it is what a
    two-stage analysis uses, and Sec. 9.1 lists the shortcomings of that
    route -- time points with too little data drop out entirely, and
    combining the per-time estimates requires knowing the temporal
    correlation between the statistics.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    times = np.asarray(times, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    sel = np.abs(times - float(at_time)) <= float(tol)
    if sel.sum() < 2:
        raise ValueError(f"fewer than two observations at time {at_time}")
    c, y = coords[sel], z[sel]
    i, j = np.triu_indices(y.size, k=1)
    d = np.linalg.norm(c[i] - c[j], axis=1)
    sq = (y[i] - y[j]) ** 2
    if max_dist is None:
        max_dist = float(d.max()) / 2.0 if d.size else 1.0
    keep = d <= max_dist
    d, sq = d[keep], sq[keep]
    edges = np.linspace(0.0, max_dist, int(n_bins) + 1)
    bi = np.clip(np.digitize(d, edges) - 1, 0, int(n_bins) - 1)
    counts = np.zeros(int(n_bins), dtype=int)
    total = np.zeros(int(n_bins), dtype=float)
    np.add.at(counts, bi, 1)
    np.add.at(total, bi, sq)
    gamma = np.full(int(n_bins), np.nan)
    nz = counts > 0
    gamma[nz] = total[nz] / (2.0 * counts[nz])
    return {"gamma": gamma, "counts": counts, "n_at_time": int(sel.sum()),
            "lags": 0.5 * (edges[:-1] + edges[1:]), "edges": edges}


def st_wls_objective(emp, model_fn):
    """The weighted least squares criterion of Sec. 9.4.

        sum_j sum_l  |N(h_j,k_l)| / (2 gamma(h_j,k_l;theta)^2)
                     x {gamma_hat(h_j,k_l) - gamma(h_j,k_l;theta)}^2

    Cressie's weights: pair count over twice the squared MODEL value, so
    well-populated lags and lags where the model is small both count for
    more. Empty cells contribute nothing -- they carry no information, and
    treating a NaN as a zero residual would reward a model for lags it was
    never tested at.
    """
    gamma_hat = np.asarray(emp["gamma"], dtype=float)
    counts = np.asarray(emp["counts"], dtype=float)
    hs = np.asarray(emp["space_lags"], dtype=float)
    ks = np.asarray(emp["time_lags"], dtype=float)
    hh, kk = np.meshgrid(hs, ks, indexing="ij")
    model = np.asarray(model_fn(hh, kk), dtype=float)
    ok = (counts > 0) & np.isfinite(gamma_hat) & np.isfinite(model) & (model > 0)
    if not np.any(ok):
        return np.inf
    resid = gamma_hat[ok] - model[ok]
    return float(np.sum(counts[ok] / (2.0 * model[ok] ** 2) * resid**2))


# --------------------------------------------------------------------------
# Sec. 9.5 -- spatio-temporal point processes
# --------------------------------------------------------------------------

def st_intensity(points, times, region, time_interval):
    """eq (9.20) for a homogeneous process: lambda = N / (|A| |T|).

    The definition is a limit over an infinitesimal CYLINDER with base ds and
    height dt (Dorai-Raj, 2001), not over a ball in R^3 -- the same refusal to
    treat time as a third spatial coordinate that runs through the chapter.
    Under first-order stationarity in space and time (FOST) the intensity does
    not depend on s or t, and this ratio is its estimator.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    t = np.asarray(times, dtype=float).ravel()
    if pts.shape[0] != t.size:
        raise ValueError("`points` and `times` must have the same length")
    area = _region_area(region)
    t0, t1 = (float(time_interval[0]), float(time_interval[1]))
    span = t1 - t0
    if span <= 0:
        raise ValueError("`time_interval` must have positive length")
    if area <= 0:
        raise ValueError("`region` must have positive area")
    return {"intensity": t.size / (area * span), "n": int(t.size),
            "area": area, "duration": span, "volume": area * span}


def st_marginal_intensities(points, times, region, time_interval,
                            n_space_bins=4, n_time_bins=4):
    """eqs (9.21) and (9.22), the marginal spatial and temporal intensities.

        lambda(s, .) = integral_T lambda(s, v) dv
        lambda(., t) = integral_D lambda(u, t) du

    Estimated by binning: the marginal spatial intensity on a grid cell is
    that cell's count divided by its AREA (already integrated over all of T),
    and the marginal temporal intensity in a time bin is its count divided by
    the bin WIDTH (already integrated over all of D).

    The corollaries in Sec. 9.5.3 give the checks: under first-order
    stationarity in time, lambda(s, .) = |T| lambda**(s); under first-order
    stationarity in space, lambda(., t) = |A| lambda*(t).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    t = np.asarray(times, dtype=float).ravel()
    (x0, x1, y0, y1) = _region_box(region)
    t0, t1 = float(time_interval[0]), float(time_interval[1])

    xe = np.linspace(x0, x1, int(n_space_bins) + 1)
    ye = np.linspace(y0, y1, int(n_space_bins) + 1)
    te = np.linspace(t0, t1, int(n_time_bins) + 1)
    cell_area = (xe[1] - xe[0]) * (ye[1] - ye[0])
    bin_width = te[1] - te[0]

    xi = np.clip(np.digitize(pts[:, 0], xe) - 1, 0, int(n_space_bins) - 1)
    yi = np.clip(np.digitize(pts[:, 1], ye) - 1, 0, int(n_space_bins) - 1)
    ti = np.clip(np.digitize(t, te) - 1, 0, int(n_time_bins) - 1)

    spatial = np.zeros((int(n_space_bins), int(n_space_bins)), dtype=float)
    np.add.at(spatial, (xi, yi), 1.0)
    temporal = np.zeros(int(n_time_bins), dtype=float)
    np.add.at(temporal, ti, 1.0)

    return {"marginal_spatial": spatial / cell_area,
            "marginal_temporal": temporal / bin_width,
            "cell_area": cell_area, "bin_width": bin_width,
            "x_edges": xe, "y_edges": ye, "t_edges": te}


def cstr_reference(area, duration, lam):
    """The completely spatio-temporally random (CSTR) benchmark, Sec. 9.5.3.

    A CSTR process is Poisson in BOTH space and time, so
    N(A, T) ~ Poisson(lambda |A x T|), lambda(s,t) = lambda, and the
    second-order intensity is lambda^2. The text's judgement is worth
    keeping in view: "If the CSR process is an unattainable standard for
    spatial point processes, then the CSTR process is even more so" -- its
    role is as the initial benchmark to test against, not as a plausible
    model of anything.
    """
    area = float(area)
    duration = float(duration)
    lam = float(lam)
    if area <= 0 or duration <= 0 or lam < 0:
        raise ValueError("`area`, `duration` must be positive and `lam` >= 0")
    mean = lam * area * duration
    return {"expected_count": mean, "variance": mean,
            "intensity": lam, "second_order_intensity": lam**2,
            "volume": area * duration}


def cstr_test(points, times, region, time_interval, n_space_bins=3,
              n_time_bins=3):
    """Test an observed pattern against CSTR by cell counts.

    IMPORTANT ON PROVENANCE. Sec. 9.5.3 defines the CSTR process and says its
    purpose is "to serve as the initial benchmark against which observed
    spatio-temporal patterns are tested, in much the same way as observed
    spatial point patterns are tested against CSR" -- but Chapter 9 gives NO
    test. The test used here is the book's own quadrat statistic for CSR,
    Sec. 3.3 eq (3.3),

        X^2 = sum_i sum_j (n_ij - nbar)^2 / nbar,   reference chi^2_{rc-1},

    for which the text gives the alternative expression X^2 = (rc-1)s^2/nbar
    with s^2 the SAMPLE variance of the quadrat counts -- which is what is
    computed here, with ddof = 1 and rc - 1 degrees of freedom. The book notes
    that no degree of freedom is lost estimating nbar, "since nbar is known in
    a mapped point pattern", and that X^2 "is thus also referred to as the
    index of dispersion"; Diggle (1983, p. 33) reserves that name for
    I = X^2/(rc-1). Fisher, Thornton and MacKenzie (1922) used (3.3) in this
    Poisson sense.

    The extension from quadrats in D to cells in D x T is the analogy Sec.
    9.5.3 itself draws, but it IS an extension, and is labelled as one rather
    than presented as a Chapter 9 result.

    Reported with the observed dispersion index rather than only a verdict,
    because with few cells the test has little power and a non-rejection is
    not evidence of randomness.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    t = np.asarray(times, dtype=float).ravel()
    (x0, x1, y0, y1) = _region_box(region)
    t0, t1 = float(time_interval[0]), float(time_interval[1])
    xe = np.linspace(x0, x1, int(n_space_bins) + 1)
    ye = np.linspace(y0, y1, int(n_space_bins) + 1)
    te = np.linspace(t0, t1, int(n_time_bins) + 1)
    xi = np.clip(np.digitize(pts[:, 0], xe) - 1, 0, int(n_space_bins) - 1)
    yi = np.clip(np.digitize(pts[:, 1], ye) - 1, 0, int(n_space_bins) - 1)
    ti = np.clip(np.digitize(t, te) - 1, 0, int(n_time_bins) - 1)
    counts = np.zeros((int(n_space_bins), int(n_space_bins), int(n_time_bins)),
                      dtype=float)
    np.add.at(counts, (xi, yi, ti), 1.0)
    flat = counts.ravel()
    m = flat.size
    mean = float(flat.mean())
    if mean <= 0:
        return {"index_of_dispersion": np.nan, "df": m - 1, "p_value": np.nan,
                "counts": counts, "mean_count": mean}
    var = float(flat.var(ddof=1))
    idx = (m - 1) * var / mean
    return {"index_of_dispersion": float(idx), "df": int(m - 1),
            "p_value": float(_chi2_sf(idx, m - 1)),
            "counts": counts, "mean_count": mean, "var_count": var}


def _chi2_sf(x, df):
    """P(chi^2_df > x) = Q(df/2, x/2), the regularised upper incomplete gamma.

    The HALVING of x is the whole content of the mapping and is easy to drop:
    Q(df/2, x) is a perfectly well-behaved number, just not this one. At
    df = 1, x = 3.841459 the correct value is 0.05 and the unhalved version
    returns 0.00557 -- a difference that silently turns a Poisson pattern into
    a significant departure from complete randomness.
    """
    from math import lgamma, exp, log
    x = 0.5 * float(x)
    a = 0.5 * float(df)
    if x <= 0:
        return 1.0
    if x < a + 1.0:                                  # series for P(a, x)
        term = 1.0 / a
        total = term
        n = 0
        while n < 10000:
            n += 1
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return float(1.0 - total * exp(-x + a * log(x) - lgamma(a)))
    # continued fraction for Q(a, x), Lentz
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    hh = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        hh *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return float(exp(-x + a * log(x) - lgamma(a)) * hh)


# --------------------------------------------------------------------------
# small region helpers (kept local so this module stands alone)
# --------------------------------------------------------------------------

def _region_box(region):
    r = np.asarray(region, dtype=float).ravel()
    if r.size != 4:
        raise ValueError("`region` must be (xmin, xmax, ymin, ymax)")
    x0, x1, y0, y1 = r
    if x1 <= x0 or y1 <= y0:
        raise ValueError("`region` must have positive extent")
    return x0, x1, y0, y1


def _region_area(region):
    x0, x1, y0, y1 = _region_box(region)
    return (x1 - x0) * (y1 - y0)
