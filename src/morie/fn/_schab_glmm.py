# SPDX-License-Identifier: AGPL-3.0-or-later
"""Spatial GLMs, GLMMs, and the CAR family of disease-mapping priors.

Two primary sources, because one book does not cover the whole family.

Schabenberger, O. & Gotway, C. A. (2005), *Statistical Methods for Spatial
Data Analysis*, Sec. 6.3-6.4, supplies the conditional specification
(6.73)-(6.74), the pseudo-likelihood machinery (6.78)-(6.85), prediction
(6.87)-(6.91), and the disease-mapping hierarchy (6.99)-(6.104).

Besag, J., York, J. & Mollie, A. (1991), "Bayesian Image Restoration, with
Two Applications in Spatial Statistics", *Annals of the Institute of
Statistical Mathematics* 43(1):1-59, Sec. 4, is the primary source for the
convolution model: the intrinsic autoregression (4.2) with its conditional
moments (4.3), the median-based alternative (4.4), the joint posterior
(4.5), and the hyperprior (4.6). Schabenberger cites this paper but never
states the convolution, so implementing `bym_*` from the textbook alone
would have meant inventing it.

Tonui, B., Mwalili, S. & Wanjoya, A. (2018), "Spatio-Temporal Variation of
HIV Infection in Kenya", *Open Journal of Statistics* 8:811-830, supplies
the matrix form of the ICAR structure (their eqs 3-5), the Leroux LCAR
prior (6)-(7), the random-walk temporal priors, the Kronecker space-time
interaction structures with their rank deficiencies (Table 1), and the
null-space constraint (12) that restores identifiability.

Everything here is internal.
"""

from . import _array_core as np

__all__ = []


# --------------------------------------------------------------------------
# link and variance functions
# --------------------------------------------------------------------------

_LINKS = ("log", "logit", "identity")
_FAMILIES = ("poisson", "binomial", "gaussian")


def link(eta_or_mu, kind, inverse=False):
    """g(mu) or its inverse g^{-1}(eta)."""
    x = np.asarray(eta_or_mu, dtype=float)
    if kind == "log":
        return np.exp(x) if inverse else np.log(x)
    if kind == "logit":
        if inverse:
            return 1.0 / (1.0 + np.exp(-x))
        return np.log(x / (1.0 - x))
    if kind == "identity":
        return x
    raise ValueError(f"`link` must be one of {_LINKS}, got {kind!r}")


def link_derivative(mu, kind):
    """g'(mu) = d eta / d mu, the multiplier in the pseudo-data (6.78)."""
    mu = np.asarray(mu, dtype=float)
    if kind == "log":
        return 1.0 / mu
    if kind == "logit":
        return 1.0 / (mu * (1.0 - mu))
    if kind == "identity":
        return np.ones_like(mu)
    raise ValueError(f"`link` must be one of {_LINKS}, got {kind!r}")


def mu_eta(mu, kind):
    """d mu / d eta, the diagonal of Psi in Sec. 6.3.5.

    Psi is defined with typical element [d mu(s_i) / d eta(s_i)], and
    g'(mu) is its reciprocal -- the text notes this explicitly when
    deriving (6.89). Keeping both spellings avoids inverting twice.
    """
    return 1.0 / link_derivative(mu, kind)


def variance_function(mu, family):
    """v(mu) in Var[Z|S] = sigma^2 v(mu), eq (6.74)."""
    mu = np.asarray(mu, dtype=float)
    if family == "poisson":
        return mu
    if family == "binomial":
        return mu * (1.0 - mu)
    if family == "gaussian":
        return np.ones_like(mu)
    raise ValueError(f"`family` must be one of {_FAMILIES}, got {family!r}")


def canonical_link(family):
    return {"poisson": "log", "binomial": "logit", "gaussian": "identity"}[family]


# --------------------------------------------------------------------------
# Sec. 6.3.4 -- the conditional specification
# --------------------------------------------------------------------------

def conditional_mean(X, beta, S, link_kind="log"):
    """eq (6.73): g[mu(s)] = x(s)'beta + S(s), so mu = g^{-1}(X beta + S).

    S(s) is a random addition to the intercept -- a random intercept that
    varies with spatial location.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    eta = X @ np.asarray(beta, dtype=float).ravel() + np.asarray(S, dtype=float).ravel()
    return link(eta, link_kind, inverse=True)


def conditional_variance(mu, sigma2, family):
    """eq (6.74): Var[Z(s)|S] = sigma^2 v(mu(s))."""
    return float(sigma2) * variance_function(mu, family)


def marginal_moments_lognormal(X, beta, sigma2_S, sigma2=1.0, rho=None):
    """Example 6.6 -- the marginal moments under a log link.

    With m(s) = exp{x(s)'beta}, canonical log link and v(mu) = mu,

        E[Z(s)]   = m(s) exp{sigma_S^2 / 2}
        Var[Z(s)] = m(s) sigma^2 exp{sigma_S^2 / 2}
                    + m(s)^2 exp{sigma_S^2}(exp{sigma_S^2} - 1)
        Cov[Z(si), Z(sj)] = m(si) m(sj) exp{sigma_S^2}
                            (exp{sigma_S^2 rho(si - sj)} - 1)

    NOTE ON THE SECOND VARIANCE TERM. The printed text renders it with
    m(s) rather than m(s)^2. It must be the square: the variance is
    E[Var(Z|S)] + Var(E[Z|S]), whose second part is m(s)^2 Var[exp{S}],
    and the book's own covariance expression on the following line reduces
    to exactly that at i = j, where rho = 1. The squared form is used here
    and the covariance identity is asserted in the tests, so the two cannot
    silently disagree.

    This function exists because of the trap Sec. 6.3.4 spells out:
    E[Z(s)] is NOT g^{-1}(x(s)'beta). Taking expectations does not carry
    through a nonlinear link, so evaluating the inverse link at the linear
    predictor gives the median-ish conditional quantity, not the marginal
    mean. `naive_marginal_mean` below returns the wrong answer on purpose,
    so callers can see the size of the gap.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m = np.exp(X @ np.asarray(beta, dtype=float).ravel())
    s2 = float(sigma2_S)
    mean = m * np.exp(s2 / 2.0)
    var = m * float(sigma2) * np.exp(s2 / 2.0) + m**2 * np.exp(s2) * (np.exp(s2) - 1.0)
    out = {"mean": mean, "variance": var, "m": m}
    if rho is not None:
        r = np.asarray(rho, dtype=float)
        out["covariance"] = np.outer(m, m) * np.exp(s2) * (np.exp(s2 * r) - 1.0)
    return out


def naive_marginal_mean(X, beta, link_kind="log"):
    """g^{-1}(x(s)'beta) -- what the marginal mean is NOT, in a GLMM.

    Provided so the discrepancy against `marginal_moments_lognormal` can be
    measured rather than assumed away. Under a log link the ratio is
    exp{sigma_S^2 / 2}, so the naive value understates the marginal mean by
    a factor that grows with the variance of the latent field.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return link(X @ np.asarray(beta, dtype=float).ravel(), link_kind, inverse=True)


# --------------------------------------------------------------------------
# Sec. 6.3.5 -- pseudo-likelihood / penalized quasi-likelihood
# --------------------------------------------------------------------------

def pseudo_data(z, mu, link_kind):
    """eq (6.78): nu_i = g(mu_i) + g'(mu_i)(Z(s_i) - mu_i)."""
    z = np.asarray(z, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    return link(mu, link_kind) + link_derivative(mu, link_kind) * (z - mu)


def sigma_mu(mu, sigma2, family, link_kind, R=None):
    """eq (6.79): Sigma_mu = sigma^2 Psi^-1 V^1/2 R(theta) V^1/2 Psi^-1.

    R is the working correlation of the conditional (residual) part. The
    text's own guidance on where to put the spatial dependence: a MARGINAL
    model lets R be a spatial correlation matrix and sets S = 0; a
    CONDITIONAL model puts the dependence in Sigma_S and takes R = I. R
    defaults to the identity, which is the conditional case.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    n = mu.size
    psi_inv = link_derivative(mu, link_kind)          # Psi^-1 = g'(mu)
    v_half = np.sqrt(variance_function(mu, family))
    if R is None:
        R = np.eye(n)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    d = psi_inv * v_half
    return float(sigma2) * (d[:, None] * R * d[None, :])


def data_covariance(mu, sigma2, family, R=None):
    """sigma^2 V^{1/2} R V^{1/2} -- the covariance on the DATA scale.

    Distinct from `sigma_mu`, eq (6.79), which is the covariance of the
    PSEUDO-data and therefore carries Psi^-1 on both sides. Sec. 6.3.5.3
    writes its score equations with the symbol Sigma_mu, but the matrix they
    need is this one: taken literally with (6.79) the scores are wrong by a
    factor of Psi^2, and they do not vanish at the pseudo-likelihood
    solution. See `pql_score`.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    n = mu.size
    v_half = np.sqrt(variance_function(mu, family))
    if R is None:
        R = np.eye(n)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    return float(sigma2) * (v_half[:, None] * R * v_half[None, :])


def gls_beta(X, Sigma_nu, nu):
    """eq (6.80): beta_hat = (X' Sigma_nu^-1 X)^-1 X' Sigma_nu^-1 nu."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    nu = np.asarray(nu, dtype=float).ravel()
    sinv = np.linalg.inv(np.atleast_2d(np.asarray(Sigma_nu, dtype=float)))
    xsx = X.T @ sinv @ X
    return np.linalg.solve(xsx, X.T @ sinv @ nu), np.linalg.inv(xsx)


def predict_random_field(Sigma_S, Sigma_nu, nu, X, beta):
    """eq (6.81): S_hat = Sigma_S Sigma_nu^-1 (nu - X beta_hat)."""
    Sigma_S = np.atleast_2d(np.asarray(Sigma_S, dtype=float))
    sinv = np.linalg.inv(np.atleast_2d(np.asarray(Sigma_nu, dtype=float)))
    resid = np.asarray(nu, dtype=float).ravel() - np.atleast_2d(X) @ np.asarray(beta).ravel()
    return Sigma_S @ sinv @ resid


def reml_objective(X, Sigma_nu, nu):
    """eq (6.84), minus twice the restricted log likelihood of the pseudo-model.

        phi_R = ln|Sigma_nu| + ln|X' Sigma_nu^-1 X| + r' Sigma_nu^-1 r
                + (n - k) ln(2 pi)

    with r the GLS residual. Same K-free Harville form used for the
    Gaussian case in Sec. 5.5.3, applied here to the linearised pseudo-data.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    nu = np.asarray(nu, dtype=float).ravel()
    S = np.atleast_2d(np.asarray(Sigma_nu, dtype=float))
    n, k = X.shape
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return np.inf
    sinv = np.linalg.inv(S)
    xsx = X.T @ sinv @ X
    sign2, logdet_xsx = np.linalg.slogdet(xsx)
    if sign2 <= 0:
        return np.inf
    beta = np.linalg.solve(xsx, X.T @ sinv @ nu)
    r = nu - X @ beta
    return float(logdet + logdet_xsx + r @ sinv @ r + (n - k) * np.log(2.0 * np.pi))


def fit_pseudo_likelihood(z, X, Sigma_S, family="poisson", link_kind=None,
                          sigma2=1.0, R=None, max_iter=100, tol=1e-8):
    """The six-step algorithm of Sec. 6.3.5.2, verbatim.

    1. initial mu from the non-spatial GLM;
    2. pseudo-data from (6.78);
    3. estimate covariance parameters from the pseudo-data (here Sigma_S is
       supplied, so this step is the caller's);
    4. GLS for beta (6.80), sigma^2 (6.82), and S_hat (6.81);
    5. update mu = g^{-1}(X beta_hat + S_hat);
    6. repeat to convergence.

    Wolfinger and O'Connell (1993). Because the expansion locus moves each
    round, this is a doubly iterative scheme: an inner linear-mixed-model
    solve inside an outer relinearisation.
    """
    z = np.asarray(z, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Sigma_S = np.atleast_2d(np.asarray(Sigma_S, dtype=float))
    if link_kind is None:
        link_kind = canonical_link(family)
    n, k = X.shape
    if z.size != n or Sigma_S.shape != (n, n):
        raise ValueError("`z`, `X` and `Sigma_S` must agree on the sample size")

    mu = _initial_mu(z, family)                       # step 1
    beta = np.zeros(k)
    S_hat = np.zeros(n)
    converged = False
    for it in range(int(max_iter)):
        nu = pseudo_data(z, mu, link_kind)            # step 2
        Sig_mu = sigma_mu(mu, sigma2, family, link_kind, R=R)
        Sigma_nu = Sigma_S + Sig_mu                   # Var[nu], Sec. 6.3.5.2
        beta_new, cov_beta = gls_beta(X, Sigma_nu, nu)          # step 4
        S_new = predict_random_field(Sigma_S, Sigma_nu, nu, X, beta_new)
        resid = nu - X @ beta_new
        sigma2_hat = float(resid @ np.linalg.solve(Sigma_nu, resid) / n)   # (6.82)
        mu_new = link(X @ beta_new + S_new, link_kind, inverse=True)  # step 5
        delta = max(np.max(np.abs(beta_new - beta)),
                    np.max(np.abs(S_new - S_hat)))
        beta, S_hat, mu = beta_new, S_new, mu_new
        if delta < tol:
            converged = True
            break

    nu = pseudo_data(z, mu, link_kind)
    Sigma_nu = Sigma_S + sigma_mu(mu, sigma2, family, link_kind, R=R)
    _, cov_beta = gls_beta(X, Sigma_nu, nu)
    return {"beta": beta, "S": S_hat, "mu": mu, "sigma2": sigma2_hat,
            "cov_beta": cov_beta,                       # Var(beta) = (X'Sig^-1X)^-1
            "se_beta": np.sqrt(np.diag(cov_beta)),
            "Sigma_nu": Sigma_nu, "pseudo_data": nu,
            "n_iter": it + 1, "converged": converged,
            "link": link_kind, "family": family}


def _initial_mu(z, family):
    """Step 1: a starting mu from the data, as the text suggests."""
    z = np.asarray(z, dtype=float).ravel()
    if family == "poisson":
        return np.maximum(z, 0.25)              # keep the log link finite
    if family == "binomial":
        return np.clip(z, 1e-3, 1.0 - 1e-3)
    return z.copy()


def pql_score(z, X, beta, S, Sigma_S, family, link_kind, sigma2=1.0, R=None):
    """The first-order conditions of the Laplace/PQL problem, Sec. 6.3.5.3.

        X' Psi Sigma_mu^-1 (Z - g^{-1}(X beta + S))            = 0
        Psi Sigma_mu^-1 (Z - g^{-1}(X beta + S)) - Sigma_S^-1 S = 0

    Breslow and Clayton (1993) reach these by a Laplace approximation with
    Fisher scoring, where PL reaches (6.80)-(6.81) by relinearisation. The
    text's conclusion is worth stating plainly: the two objective functions
    "differ ... only by a constant amount. The two approaches will thus
    yield the same estimates." So this is a CHECK on `fit_pseudo_likelihood`,
    not a second answer -- the tests assert both scores vanish at its
    solution.

    ONE CORRECTION TO THE PRINTED EQUATIONS. Sec. 6.3.5.3 writes Sigma_mu in
    these score equations, but Sigma_mu was defined at (6.79) as the
    covariance of the PSEUDO-data, which carries Psi^-1 on both sides. The
    matrix the scores actually require is the DATA-scale covariance
    sigma^2 V^{1/2} R V^{1/2} -- `data_covariance` above. The symbol is
    reused for two different matrices.

    This is not a matter of taste. Stationarity of the pseudo-likelihood
    gives X' Sigma_mu^-1 (nu - X beta - S) = X' Sigma_mu^-1 Psi^-1 (Z - mu),
    and Sigma_mu^-1 Psi^-1 = Psi Sigma_data^-1 exactly. Using (6.79) as
    written instead leaves a spurious factor of Psi^2: measured on a
    40-point Poisson fixture the scores came to 9.1e+01 and 4.4e+01 rather
    than 7.9e-15 and 1.3e-14, and the ratio of the two weightings was Psi^2
    to machine precision. As a further check, for the Poisson log link
    Psi = V = mu, so Psi Sigma_data^-1 is the identity and score_beta must
    reduce to the ordinary Poisson score X'(Z - mu) -- which it does.
    """
    z = np.asarray(z, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    S = np.asarray(S, dtype=float).ravel()
    mu = link(X @ np.asarray(beta, dtype=float).ravel() + S, link_kind, inverse=True)
    psi = np.diag(mu_eta(mu, link_kind))
    sig_mu_inv = np.linalg.inv(data_covariance(mu, sigma2, family, R=R))
    resid = z - mu
    common = psi @ sig_mu_inv @ resid
    return {"score_beta": X.T @ common,
            "score_S": common - np.linalg.solve(
                np.atleast_2d(np.asarray(Sigma_S, dtype=float)), S)}


# --------------------------------------------------------------------------
# Sec. 6.3.6 -- prediction
# --------------------------------------------------------------------------

def predict_glm(nu0_hat, sigma2_nu0, mu0_hat, link_kind):
    """eqs (6.87)-(6.91).

    Two different predictors of the ORIGINAL data live here, and the book is
    emphatic that their error measures are not interchangeable.

    `inverse_link` is eq (6.87), Z_hat = g^{-1}(p(nu; nu(s0))). The delta
    method applied to it gives (6.88) -- and the text states outright that
    (6.88) "is not the mean-squared prediction error of the inverse linked
    predictor (6.87). It is the prediction error of a different predictor."

    That different predictor is eq (6.90), obtained by assuming the new
    observation also satisfies (6.78) and solving:

        Z_hat(s0) = mu_hat(s0) + (g'(mu_hat(s0)))^-1 (nu_hat(s0) - g(mu_hat(s0)))

    whose mean-squared prediction error is exactly (6.91),

        sigma^2_Z(s0) = [(g'(mu_hat(s0)))^-1]^2 sigma^2_nu(s0)
                      = (d mu / d eta)^2 sigma^2_nu(s0).

    Both are returned, each paired with the error measure that belongs to
    it, and `mspe_is_for` records which. Gotway and Wolfinger (2003).
    """
    nu0 = float(nu0_hat)
    s2 = float(sigma2_nu0)
    mu0 = float(mu0_hat)
    gprime = float(link_derivative(np.array(mu0), link_kind))
    dmu_deta = 1.0 / gprime
    linearised = mu0 + (nu0 - float(link(np.array(mu0), link_kind))) / gprime
    return {"prediction": linearised,                       # eq (6.90)
            "mspe": dmu_deta**2 * s2,                       # eq (6.91)
            "prediction_error": np.sqrt(dmu_deta**2 * s2),
            "inverse_link_prediction": float(link(np.array(nu0), link_kind,
                                                  inverse=True)),   # eq (6.87)
            "pseudo_scale_prediction": nu0,
            "pseudo_scale_mspe": s2,
            "mspe_is_for": "eq (6.90), the linearised predictor -- NOT the "
                           "inverse-link predictor of eq (6.87)"}


# --------------------------------------------------------------------------
# CAR family: structure matrices and priors
# --------------------------------------------------------------------------

def neighbour_structure(adjacency):
    """R with R_ii = n_i, R_ij = -1 if j ~ i, 0 otherwise (Tonui et al. eq 4).

    This is the ICAR structure (precision) matrix. It is singular by
    construction: rows sum to zero, so R 1 = 0 and the constant vector spans
    its null space. That is why eq (3) writes the covariance with a
    Moore-Penrose inverse rather than an ordinary one.
    """
    A = np.atleast_2d(np.asarray(adjacency, dtype=float))
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("`adjacency` must be square")
    if not np.allclose(A, A.T):
        raise ValueError("`adjacency` must be symmetric")
    if np.any(np.diag(A) != 0):
        raise ValueError("`adjacency` must have a zero diagonal (no self-neighbours)")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("`adjacency` must be a 0/1 matrix")
    return np.diag(A.sum(axis=1)) - A


def icar_covariance(R, sigma2=1.0):
    """eq (3): u ~ N(0, sigma^2 R^-), with R^- the Moore-Penrose inverse.

    The ordinary inverse does not exist. Using `np.linalg.pinv` is not a
    numerical convenience here, it is the model as written.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    return float(sigma2) * np.linalg.pinv(R)


def icar_full_conditional(u, adjacency, sigma2=1.0):
    """eq (5): u_i | u_-i ~ N( (1/n_i) sum_{j~i} u_j , sigma^2 / n_i ).

    The conditional mean is the plain average of the neighbours and the
    conditional variance shrinks with the number of neighbours -- the two
    properties that make ICAR a smoother.
    """
    u = np.asarray(u, dtype=float).ravel()
    A = np.atleast_2d(np.asarray(adjacency, dtype=float))
    n_i = A.sum(axis=1)
    if np.any(n_i == 0):
        raise ValueError("every area must have at least one neighbour")
    return {"mean": (A @ u) / n_i, "variance": float(sigma2) / n_i,
            "n_neighbours": n_i}


def lcar_precision(R, rho, sigma2=1.0):
    """eq (6): u ~ N(0, sigma^2 Q^-1) with Q = rho R + (1 - rho) I.

    Leroux, Lei and Breslow. The point of this prior is that it NESTS the
    two extremes the BYM convolution tries to mix:

        rho = 0  ->  Q = I,  the exchangeable (independent) prior
        rho = 1  ->  Q = R,  the ICAR prior

    so a single interpretable parameter in [0, 1] moves between them,
    instead of two variance components that the data cannot separate. Both
    limits are asserted in the tests.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    rho = float(rho)
    if not (0.0 <= rho <= 1.0):
        raise ValueError("`rho` must lie in [0, 1]")
    if sigma2 <= 0:
        raise ValueError("`sigma2` must be positive")
    n = R.shape[0]
    return rho * R + (1.0 - rho) * np.eye(n), float(sigma2)


def lcar_full_conditional(u, adjacency, rho, sigma2=1.0):
    """eq (7).

        u_i | u_j!=i ~ N( rho/((1-rho) + n_i rho) sum_{j~i} u_j ,
                          sigma_u^2 / ((1-rho) + n_i rho) )

    At rho = 1 this collapses to the ICAR conditional (5); at rho = 0 the
    mean is zero and the variance is sigma^2, the exchangeable case.
    """
    u = np.asarray(u, dtype=float).ravel()
    A = np.atleast_2d(np.asarray(adjacency, dtype=float))
    rho = float(rho)
    if not (0.0 <= rho <= 1.0):
        raise ValueError("`rho` must lie in [0, 1]")
    n_i = A.sum(axis=1)
    denom = (1.0 - rho) + n_i * rho
    return {"mean": rho * (A @ u) / denom, "variance": float(sigma2) / denom,
            "n_neighbours": n_i}


def bym_convolution(u, v):
    """The Besag-York-Mollie convolution: psi = u + v.

    u carries spatially structured variation under an intrinsic
    autoregression; v carries unstructured heterogeneity under a Gaussian
    exchangeable prior. Besag, York and Mollie (1991) Sec. 4 write the log
    relative risk as x = u + v and the relative risk as exp{u + v}.
    Schabenberger gives the two components separately, at (6.104) and
    (6.102), but never their sum.

    Their reading of the two scale parameters is worth carrying: "kappa
    tends to 0 implies constant u_i's, whereas kappa large implies
    correspondingly large but spatially structured variation. Similarly,
    lambda tends to 0 implies v = 0, whereas lambda large implies
    substantial but unstructured extra-Poisson variability."

    IDENTIFIABILITY. Only the sum enters the likelihood, so the data cannot
    separate sigma_u^2 from sigma_v^2 -- Tonui et al. state that "the
    variance components in the BYM convolution model are not identifiable
    from the data ... and informative hyper priors are needed for posterior
    inference." `bym_identifiability_note` returns that warning with every
    fit rather than leaving it to the reader, and `lcar_precision` is the
    principled alternative.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if u.size != v.size:
        raise ValueError("`u` and `v` must have the same length")
    return u + v


def bym_identifiability_note():
    return ("only u + v enters the likelihood, so sigma_u^2 and sigma_v^2 are "
            "not separately identifiable from the data; informative "
            "hyperpriors are required, or use the Leroux LCAR prior, which "
            "nests the exchangeable (rho=0) and ICAR (rho=1) cases in one "
            "identifiable parameter")


def smr(counts, expected):
    """Z(s_i)/E(s_i), the standardized mortality ratio.

    Sec. 6.4.3.2: the SMR is the maximum likelihood estimate of the relative
    risk zeta(s_i) under eq (6.99), and it is exactly the quantity these
    priors are meant to smooth.
    """
    z = np.asarray(counts, dtype=float).ravel()
    e = np.asarray(expected, dtype=float).ravel()
    if z.size != e.size:
        raise ValueError("`counts` and `expected` must have the same length")
    if np.any(e <= 0):
        raise ValueError("`expected` counts must be positive")
    return z / e


def poisson_disease_mean(expected, X, beta, psi):
    """eq (6.101): E[Z(s_i)|beta, psi] = E(s_i) exp{x(s_i)'beta + psi(s_i)}.

    log{E(s_i)} is an offset, not a covariate -- its coefficient is fixed at
    one (Tonui et al. eq 1: log(mu_i) = log(E_i) + log(theta_i)).
    """
    e = np.asarray(expected, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return e * np.exp(X @ np.asarray(beta, dtype=float).ravel()
                      + np.asarray(psi, dtype=float).ravel())


# --------------------------------------------------------------------------
# temporal and space-time structures (Tonui et al. Sec. 2.2, Table 1)
# --------------------------------------------------------------------------

def random_walk_structure(n_time, order=1):
    """Structure matrix of an RW1 or RW2 prior.

    RW1: gamma_t | gamma_{t-1} ~ N(gamma_{t-1}, sigma^2)
    RW2: gamma_t | gamma_{t-1}, gamma_{t-2} ~ N(2 gamma_{t-1} - gamma_{t-2}, sigma^2)

    Built as D'D from the differencing matrix, so the rank deficiency comes
    out right by construction: 1 for RW1 (the constant), 2 for RW2 (constant
    and linear trend).
    """
    T = int(n_time)
    k = int(order)
    if k not in (1, 2):
        raise ValueError("`order` must be 1 or 2")
    if T <= k:
        raise ValueError(f"need more than {k} time points for an RW{k}")
    D = np.zeros((T - k, T))
    row = {1: [-1.0, 1.0], 2: [1.0, -2.0, 1.0]}[k]
    for i in range(T - k):
        D[i, i:i + k + 1] = row
    return D.T @ D


def interaction_structure(R_space, R_time, kind):
    """Table 1: the four Knorr-Held space-time interaction structures.

        Type I    I_s (x) I_t      no structure in either margin
        Type II   I_s (x) R_t      independent time trends per area
        Type III  R_s (x) I_t      independent spatial pattern per period
        Type IV   R_s (x) R_t      dependent over both

    Returns the Kronecker product together with its rank and rank
    deficiency, because the deficiency is what determines how many
    constraints eq (12) needs.
    """
    kinds = ("I", "II", "III", "IV")
    if kind not in kinds:
        raise ValueError(f"`kind` must be one of {kinds}")
    Rs = np.atleast_2d(np.asarray(R_space, dtype=float))
    Rt = np.atleast_2d(np.asarray(R_time, dtype=float))
    ns, nt = Rs.shape[0], Rt.shape[0]
    Is, It = np.eye(ns), np.eye(nt)
    M = {"I": np.kron(Is, It), "II": np.kron(Is, Rt),
         "III": np.kron(Rs, It), "IV": np.kron(Rs, Rt)}[kind]
    rank = int(np.linalg.matrix_rank(M))
    return {"structure": M, "kind": kind, "rank": rank,
            "rank_deficiency": M.shape[0] - rank,
            "n_constraints_required": M.shape[0] - rank}


def null_space_constraints(R_delta, tol=None):
    """eq (12): pi*(delta) = pi(delta | A delta = e).

    "A [is] given by those eigenvectors of R_delta which span the null
    space. Hence, to ensure the identifiability of delta, the null space of
    the respective structure matrix R_delta is computed using the obtained
    eigenvectors as linear constraints ... the number of linear constraints
    which are necessary is always equal to the rank deficiency of R_delta
    and e will be a vector of zeros."

    Without these the interaction is confounded with the main time effect.
    Only Type I needs none, because only Type I is of full rank.
    """
    M = np.atleast_2d(np.asarray(R_delta, dtype=float))
    vals, vecs = np.linalg.eigh(M)
    scale = max(float(np.abs(vals).max()), 1.0)
    if tol is None:
        tol = 1e-10 * scale * M.shape[0]
    null = np.abs(vals) <= tol
    A = vecs[:, null].T
    return {"A": A, "e": np.zeros(A.shape[0]),
            "n_constraints": int(A.shape[0]),
            "rank_deficiency": int(null.sum())}


def apply_sum_to_zero(delta, A):
    """Project delta onto {x : A x = 0}, the constrained subspace of eq (12)."""
    d = np.asarray(delta, dtype=float).ravel()
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if A.size == 0 or A.shape[0] == 0:
        return d
    return d - A.T @ np.linalg.solve(A @ A.T, A @ d)


def linear_trend_log_risk(alpha, u, beta_t, delta_i, times):
    """eq (9): log(theta_it) = alpha + u_i + (beta + delta_i) t.

    alpha the global risk, u_i the spatial effect, beta the global linear
    time trend, delta_i the area-specific differential trend -- the
    interaction between the linear trend and the spatial effect
    (Bernardinelli et al.).
    """
    u = np.asarray(u, dtype=float).ravel()
    d = np.asarray(delta_i, dtype=float).ravel()
    t = np.asarray(times, dtype=float).ravel()
    if u.size != d.size:
        raise ValueError("`u` and `delta_i` must have one entry per area")
    return (float(alpha) + u[:, None]
            + (float(beta_t) + d[:, None]) * t[None, :])


def nonparametric_log_risk(alpha, u, phi, gamma, delta=None):
    """eq (10): log(theta_it) = alpha + u_i + phi_t + gamma_t + delta_it.

    phi_t temporally unstructured, gamma_t temporally structured (RW1/RW2),
    delta_it the space-time interaction. Dropping delta gives the additive
    model.
    """
    u = np.asarray(u, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    gam = np.asarray(gamma, dtype=float).ravel()
    if phi.size != gam.size:
        raise ValueError("`phi` and `gamma` must have one entry per time point")
    out = float(alpha) + u[:, None] + (phi + gam)[None, :]
    if delta is not None:
        D = np.asarray(delta, dtype=float)
        if D.shape != out.shape:
            raise ValueError(f"`delta` must be shaped {out.shape}")
        out = out + D
    return out

# --------------------------------------------------------------------------
# Besag, York & Mollie (1991) Sec. 4 -- the convolution model itself
# --------------------------------------------------------------------------

def bym_icar_log_prior(u, adjacency, kappa):
    """eq (4.2): p(u | kappa) ~ kappa^{-n/2} exp{-(1/2 kappa) sum_{i~j}(u_i-u_j)^2}.

    A Gaussian intrinsic autoregression. The pairwise sum is u' R u with R
    the structure matrix, so the implied precision is R / kappa. The paper
    is explicit that this "is strictly improper because it only addresses
    differences in the u_i's and not their overall level" -- R has zero row
    sums, so adding a constant to every u_i leaves the density unchanged.

    Returns the log density up to the additive constant.
    """
    u = np.asarray(u, dtype=float).ravel()
    R = neighbour_structure(adjacency)
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError("`kappa` must be positive")
    n = u.size
    return -0.5 * n * np.log(kappa) - float(u @ R @ u) / (2.0 * kappa)


def bym_icar_conditional_moments(u, adjacency, kappa):
    """eq (4.3): E(u_i | ...) = ubar_i, Var(u_i | ...) = kappa / n_i.

    Identical in form to `icar_full_conditional` with sigma^2 = kappa; kept
    under the BYM name because the paper's kappa is the scale that appears
    in (4.2) and (4.5), and conflating the two symbols is how sign and
    scaling errors get in.
    """
    return icar_full_conditional(u, adjacency, sigma2=kappa)


def bym_median_log_prior(u, adjacency, kappa):
    """eq (4.4), the alternative with phi(z) = |z| / kappa.

        p_i(u_i | ...) ~ (1/kappa) exp{-(1/kappa) sum_{j in di} |u_i - u_j|}

    Its mode is at the MEDIAN rather than the mean of the contiguous u_i's,
    which the paper argues is "more appropriate than (4.2) if
    discontinuities in the risk surface are expected" -- a stochastic
    version of the median filter. Sec. 4 reports the estimated risks were
    "almost identical when the prior (4.2) was replaced by (4.4)" on their
    data, so the choice is not always consequential.
    """
    u = np.asarray(u, dtype=float).ravel()
    A = np.atleast_2d(np.asarray(adjacency, dtype=float))
    kappa = float(kappa)
    if kappa <= 0:
        raise ValueError("`kappa` must be positive")
    i, j = np.triu_indices(u.size, k=1)
    pairs = A[i, j] > 0
    total = float(np.sum(np.abs(u[i][pairs] - u[j][pairs])))
    return -u.size * np.log(kappa) - total / kappa


def bym_log_posterior(y, c, u, v, kappa, lam, adjacency, epsilon=0.01):
    """eq (4.5) with the hyperprior (4.6), up to an additive constant.

        P(u,v,kappa,lambda | y) ~ prod_i {exp(-c_i e^{x_i}) (c_i e^{x_i})^{y_i} / y_i!}
            x kappa^{-n/2} exp{-(1/2 kappa) sum_{i~j} (u_i - u_j)^2}
            x lambda^{-n/2} exp{-(1/2 lambda) sum_i v_i^2}
            x prior(kappa, lambda)

    with x_i = u_i + v_i, and (4.6) prior(kappa, lambda) ~
    exp{-eps/2kappa} exp{-eps/2lambda}, eps = 0.01.

    The reason (4.6) exists is worth recording. The "obvious choice"
    proportional to kappa^-1 lambda^-1 leaves (4.5) improper near the
    origin, and the paper notes this "does not stem from any spatial
    aspects of the formulation but is a common and unpleasant feature of
    Bayesian hierarchical models in general". Even once proper, a
    singularity at the origin "invalidates the Gibbs sampler, because the
    origin becomes an absorbing state of the Markov chain". eps is the fix.
    """
    y = np.asarray(y, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if not (y.size == c.size == u.size == v.size):
        raise ValueError("`y`, `c`, `u` and `v` must have the same length")
    if np.any(c <= 0):
        raise ValueError("`c` (expected counts) must be positive")
    kappa, lam, eps = float(kappa), float(lam), float(epsilon)
    if kappa <= 0 or lam <= 0:
        raise ValueError("`kappa` and `lam` must be positive")
    n = y.size
    x = u + v
    loglik = float(np.sum(-c * np.exp(x) + y * (np.log(c) + x)))
    return (loglik
            + bym_icar_log_prior(u, adjacency, kappa)
            - 0.5 * n * np.log(lam) - float(v @ v) / (2.0 * lam)
            - eps / (2.0 * kappa) - eps / (2.0 * lam))


def bym_map(y, c, adjacency, kappa, lam, max_iter=200, tol=1e-11):
    """Conditional MAP estimates u*, v* given kappa and lambda, Sec. 4.

    Maximises the log of (4.5) over (u, v). The paper guarantees this is
    well posed: "the logarithm of the joint posterior density of u and v,
    given kappa, lambda and y, is a strictly concave differentiable function
    of u and v and therefore possesses a single maximum". So Newton's method
    converges to the global optimum and there is no multi-start question.

    Gradients, with x = u + v:
        dL/du = y - c e^x - R u / kappa
        dL/dv = y - c e^x - v / lambda
    Hessian blocks: -diag(c e^x) - R/kappa, -diag(c e^x) - I/lambda, and
    -diag(c e^x) off-diagonal; negative definite, hence the concavity.

    TWO IDENTITIES COME FREE, and the paper states both as properties of
    u*, v*: "sum_i v*_i = 0" and "sum_i c_i e^{u*_i + v*_i} = sum_i y_i, so
    that the fitted total number of cases matches the observed total".

    They are consequences of stationarity rather than imposed constraints.
    Summing dL/du over i annihilates the ICAR term because R has zero row
    sums, leaving sum(y - c e^x) = 0; substituting that into the sum of
    dL/dv leaves sum(v) = 0. Both are asserted in the tests, which makes
    them a genuine check that a true stationary point was reached rather
    than a decorative post-condition.
    """
    y = np.asarray(y, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    R = neighbour_structure(adjacency)
    n = y.size
    if c.size != n or R.shape[0] != n:
        raise ValueError("`y`, `c` and `adjacency` must agree on the number of areas")
    if np.any(c <= 0):
        raise ValueError("`c` (expected counts) must be positive")
    kappa, lam = float(kappa), float(lam)
    if kappa <= 0 or lam <= 0:
        raise ValueError("`kappa` and `lam` must be positive")

    u = np.zeros(n)
    v = np.zeros(n)
    I = np.eye(n)
    converged = False
    for it in range(int(max_iter)):
        w = c * np.exp(u + v)
        g_u = y - w - R @ u / kappa
        g_v = y - w - v / lam
        H = np.block([[-np.diag(w) - R / kappa, -np.diag(w)],
                      [-np.diag(w), -np.diag(w) - I / lam]])
        step = np.linalg.solve(H, np.concatenate([g_u, g_v]))
        u_new, v_new = u - step[:n], v - step[n:]
        delta = float(np.max(np.abs(np.concatenate([u_new - u, v_new - v]))))
        u, v = u_new, v_new
        if delta < tol:
            converged = True
            break

    x = u + v
    fitted = c * np.exp(x)
    return {"u": u, "v": v, "x": x, "relative_risk": np.exp(x),
            "fitted": fitted, "n_iter": it + 1, "converged": converged,
            "sum_v": float(v.sum()),
            "fitted_total": float(fitted.sum()),
            "observed_total": float(y.sum()),
            "log_posterior": bym_log_posterior(y, c, u, v, kappa, lam,
                                               adjacency)}
