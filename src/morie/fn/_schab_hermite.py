# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chebyshev-Hermite polynomials and disjunctive kriging.

Schabenberger & Gotway (2005), Sec. 5.6.4. Disjunctive kriging (Matheron,
1976) predicts g(Z(s0)) by expanding g in a basis that is orthonormal under
the standard Gaussian density, so the components can be kriged one at a
time without modelling cross-covariances.

The basis is the Chebyshev-Hermite system, eq (5.64),

    H_p(x) = (-1)^p exp(x^2/2) d^p/dx^p exp(-x^2/2),

built here from the three-term recurrence the text gives,

    H_0 = 1,  H_1 = x,  H_{p+1}(x) = x H_p(x) - p H_{p-1}(x),

rather than by differentiating. These are orthogonal but NOT orthonormal --
the text is explicit that the integral is p! rather than 1 -- so the
standardised eta_p(x) = H_p(x)/sqrt(p!) is what forms the orthonormal basis
and what everything downstream uses.

The coefficients are eq (5.65), b_p = integral g(x) eta_p(x) f(x) dx.

Matheron's result is what makes the scheme work: for bivariate Gaussian
Z with correlation rho(h) and p >= 1,

    Cov[eta_p(Z(s+h)), eta_p(Z(s))] = rho(h)^p,

so each component is kriged with the SAME correlation function raised to
the p-th power. Because eta_p has mean zero and unit variance the system
(5.68) is a simple-kriging system with no unbiasedness constraint,

    lambda R = rho,   R = [rho_ij^p],  rho = [rho_0i^p],

with variance (5.69) sigma^2_eta = 1 - lambda' rho, predictor (5.70) and
prediction variance (5.71) sigma^2_dk = sum_{p>=1} b_p^2 sigma^2_eta.

The text notes rho(h)^p tends to white noise as p grows, so "in practice
only a few (usually less than a dozen ...) Hermite polynomials need to be
predicted".

References
----------
Schabenberger, O. & Gotway, C. A. (2005) *Statistical Methods for
Spatial Data Analysis*, Texts in Statistical Science, Chapman &
Hall/CRC, Boca Raton, ISBN 1-58488-322-7.
Sec. 5.6 and eq (5.71).

Everything here is internal.
"""

from . import _array_core as np

__all__ = []


def hermite_e(x, degree):
    """H_0..H_degree evaluated at x, by the recurrence of Sec. 5.6.4.1.

    Returns an array of shape (degree + 1,) + x.shape.
    """
    x = np.asarray(x, dtype=float)
    degree = int(degree)
    if degree < 0:
        raise ValueError("`degree` must be non-negative")
    out = np.empty((degree + 1,) + x.shape, dtype=float)
    out[0] = 1.0
    if degree >= 1:
        out[1] = x
    for p in range(1, degree):
        out[p + 1] = x * out[p] - p * out[p - 1]
    return out


def hermite_orthonormal(x, degree):
    """eta_p(x) = H_p(x)/sqrt(p!), the orthonormal system."""
    h = hermite_e(x, degree)
    scale = np.sqrt(np.array([_factorial(p) for p in range(degree + 1)],
                             dtype=float))
    return h / scale.reshape((-1,) + (1,) * (h.ndim - 1))


def _factorial(p):
    out = 1.0
    for k in range(2, int(p) + 1):
        out *= k
    return out


def gauss_hermite(n):
    """Nodes and weights for integration against the STANDARD GAUSSIAN.

    Golub-Welsch: the nodes are the eigenvalues of the symmetric tridiagonal
    Jacobi matrix of the orthogonal system, which for the probabilists'
    Hermite polynomials has zero diagonal and sqrt(k) off-diagonal; the
    weights are the squared first components of the eigenvectors. Written
    out rather than taken from a library so the R arm runs the same
    arithmetic -- both languages then need only a symmetric eigensolver.

    The weights sum to 1, so sum(w * g(x)) approximates E[g(X)] for
    X ~ N(0, 1) directly, with no 1/sqrt(2 pi) left over.
    """
    n = int(n)
    if n < 1:
        raise ValueError("`n` must be positive")
    off = np.sqrt(np.arange(1.0, n))
    jac = np.diag(off, 1) + np.diag(off, -1)
    vals, vecs = np.linalg.eigh(jac)
    weights = vecs[0, :] ** 2
    return vals, weights


def hermite_coefficients(g, degree, n_quad=None):
    """b_p = integral g(x) eta_p(x) f(x) dx, eq (5.65).

    Evaluated by Gauss-Hermite quadrature, which is exact for polynomial g
    of degree below 2 * n_quad -- so the Example 5.12 identities come out
    exactly, not approximately.
    """
    degree = int(degree)
    if n_quad is None:
        n_quad = 2 * degree + 32
    nodes, weights = gauss_hermite(n_quad)
    eta = hermite_orthonormal(nodes, degree)
    gvals = np.asarray(g(nodes), dtype=float)
    return eta @ (weights * gvals)


def disjunctive_kriging(coords, y, target, correlation_fn, g, degree=8,
                        n_quad=None):
    """Predict g(Z(s0)) by disjunctive kriging, Sec. 5.6.4.

    `y` are the data on the GAUSSIAN scale (apply the normal-scores
    transform first if they are not). `correlation_fn` returns rho(h).

    Returns (prediction, variance, coefficients, component_variances).
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    degree = int(degree)
    if coords.shape[0] != y.size:
        raise ValueError("`coords` and `y` must have the same number of rows")

    d_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    d_vec = np.linalg.norm(coords - target, axis=1)
    rho_mat = np.asarray(correlation_fn(d_mat), dtype=float)
    rho_vec = np.asarray(correlation_fn(d_vec), dtype=float)

    b = hermite_coefficients(g, degree, n_quad=n_quad)
    eta_data = hermite_orthonormal(y, degree)

    pred = float(b[0])                     # eta_0 == 1, so b_0 enters as is
    var = 0.0
    comp_var = np.zeros(degree + 1)
    for p in range(1, degree + 1):
        r_mat = rho_mat**p
        r_vec = rho_vec**p
        try:
            lam = np.linalg.solve(r_mat, r_vec)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(r_mat, r_vec, rcond=None)[0]
        pred += float(b[p]) * float(lam @ eta_data[p])
        s2 = 1.0 - float(lam @ r_vec)      # eq (5.69)
        comp_var[p] = s2
        var += float(b[p]) ** 2 * s2       # eq (5.71)
    return pred, var, b, comp_var


def standard_normal_pdf(x):
    """f(x), the standard Gaussian density."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def standard_normal_cdf(x):
    """F(x) by the native quantile function's inverse relationship.

    Computed from the complementary error function via its series-free
    rational form is not needed here: erf is available in the standard
    library and is not a numerical dependency of the kind this package
    avoids.
    """
    from math import erf, sqrt
    x = np.asarray(x, dtype=float)
    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(x)


def indicator_coefficients(z_k, degree):
    """Exact Hermite coefficients of the indicator I(Z <= z_k), eq (5.72).

    Gauss-Hermite quadrature must NOT be used for this. It is exact for
    polynomials, and the indicator is a step function, so the quadrature
    converges slowly and silently: at degree 6 with 44 nodes it returns
    b_0 = 0.683 where F(z_k) = 0.758. Since the indicator is the canonical
    disjunctive-kriging target, that would be wrong in exactly the case the
    method exists for.

    The text derives the closed form from the identity
    H_p(x) f(x) = (-1)^p d/dx (H_{p-1}(x) f(x)), giving

        b_0 = F(z_k),
        b_p = (-1)^p H_{p-1}(z_k) f(z_k) / sqrt(p!),   p >= 1,

    so that

        I(s0, z_k) = F(z_k)
                     + f(z_k) sum_{p>=1} (-1)^p H_{p-1}(z_k) H_p(Z(s0)) / p!
    """
    degree = int(degree)
    z_k = float(z_k)
    h = hermite_e(np.array(z_k), degree)
    fz = float(standard_normal_pdf(z_k))
    b = np.empty(degree + 1)
    b[0] = float(standard_normal_cdf(z_k))
    for p in range(1, degree + 1):
        b[p] = ((-1.0) ** p) * float(h[p - 1]) * fz / np.sqrt(_factorial(p))
    return b
