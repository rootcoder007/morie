"""Shared primitives for the Morin probability shelf.

Every function here implements a named result from

    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace. Chapters 1-6.

The equation numbers in the per-module front ends refer to that book.
Counting primitives use exact integer arithmetic (math.factorial /
math.comb); probability helpers validate their inputs and fail loudly.
"""

import math

from . import _array_core as np

__all__: list = []


def _check_prob(p, name):
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")
    return p


def _check_prob_vec(ps, name):
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    if ps.ndim != 1 or ps.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if np.any(ps < 0.0) or np.any(ps > 1.0):
        raise ValueError(f"all entries of {name} must be in [0, 1]")
    return ps


def _check_nonneg_int(n, name):
    if not float(n) == int(n) or int(n) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {n!r}")
    return int(n)


# ---------------------------------------------------------------- chapter 1

def factorial(n):
    """N! -- eq (1.1)."""
    return math.factorial(_check_nonneg_int(n, "n"))


def permutations_count(n):
    """P_N = N!, permutations of N distinct objects -- eq (1.3)."""
    return math.factorial(_check_nonneg_int(n, "n"))


def partial_permutations(N, n):
    """N_P_n = N(N-1)...(N-(n-1)) = N!/(N-n)! -- eqs (1.5)-(1.6)."""
    N = _check_nonneg_int(N, "N")
    n = _check_nonneg_int(n, "n")
    if n > N:
        raise ValueError(f"n ({n}) cannot exceed N ({N})")
    prod = 1
    for j in range(N, N - n, -1):
        prod *= j
    concise = math.factorial(N) // math.factorial(N - n)
    if prod != concise:
        raise AssertionError("product and factorial forms disagree")
    return prod


def binom(N, k):
    """Binomial coefficient C(N, k), exact integer."""
    N = _check_nonneg_int(N, "N")
    k = _check_nonneg_int(k, "k")
    if k > N:
        return 0
    return math.comb(N, k)


def binomial_expansion(a, b, n):
    """Terms C(n,k) a^(n-k) b^k of the binomial theorem -- eq (1.21).

    Returns (terms, total); total equals (a + b)^n up to float rounding.
    """
    a, b = float(a), float(b)
    n = _check_nonneg_int(n, "n")
    terms = [math.comb(n, k) * a ** (n - k) * b ** k for k in range(n + 1)]
    return terms, float(sum(terms))


def hockey_stick(n, k):
    """Hockey-stick identity: sum_{j=k-1}^{n-1} C(j, k-1) = C(n, k) -- eq (1.29)."""
    n = _check_nonneg_int(n, "n")
    k = _check_nonneg_int(k, "k")
    if k < 1 or k > n:
        raise ValueError("hockey stick needs 1 <= k <= n")
    s = sum(math.comb(j, k - 1) for j in range(k - 1, n))
    return s, math.comb(n, k)


def multinomial_coefficient(ns, N=None):
    """Multinomial coefficient N!/(n1! n2! ... nk!) -- eqs (1.35), (1.37).

    If N exceeds sum(ns), the leftover people form one extra implicit
    committee of size N - sum(ns) (book's remark below eq (1.35)).
    """
    ns = [_check_nonneg_int(x, "ns[i]") for x in np.atleast_1d(ns)]
    total = sum(ns)
    if N is None:
        N = total
    N = _check_nonneg_int(N, "N")
    if total > N:
        raise ValueError(f"sum(ns) = {total} exceeds N = {N}")
    if total < N:
        ns = ns + [N - total]
    denom = 1
    for x in ns:
        denom *= math.factorial(x)
    return math.factorial(N) // denom


def stars_and_bars(n, N):
    """N_U_n = C(n + N - 1, N - 1): unordered samples with repetition -- eq (1.57)."""
    n = _check_nonneg_int(n, "n")
    N = _check_nonneg_int(N, "N")
    if N < 1:
        raise ValueError("N must be >= 1")
    return math.comb(n + N - 1, N - 1)


def sd_of_iid_sum(sigma, n):
    """sigma_sum = sqrt(n) * sigma for a sum of n i.i.d. variables -- eq (3.45)."""
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    n = _check_nonneg_int(n, "n")
    return math.sqrt(n) * sigma


# ---------------------------------------------------------------- chapter 2

def prob_and_independent(ps):
    """P(A1 and ... and Ak) = prod P(Ai) for independent events -- eqs (2.2), (2.70)."""
    ps = _check_prob_vec(ps, "ps")
    return float(np.prod(ps))


def chain_rule(p_a, p_b_given_a):
    """P(A and B) = P(A) P(B|A) -- eqs (2.5), (2.9), (2.69)."""
    return _check_prob(p_a, "p_a") * _check_prob(p_b_given_a, "p_b_given_a")


def prob_or_exclusive(ps):
    """P(A1 or ... or Ak) = sum P(Ai) for mutually exclusive events -- eq (2.14)."""
    ps = _check_prob_vec(ps, "ps")
    total = float(np.sum(ps))
    if total > 1.0 + 1e-12:
        raise ValueError("exclusive probabilities sum past 1; events not exclusive")
    return min(total, 1.0)


def prob_or_general(p_a, p_b, p_ab):
    """P(A or B) = P(A) + P(B) - P(A and B) -- eqs (2.21), (2.71)."""
    p_a = _check_prob(p_a, "p_a")
    p_b = _check_prob(p_b, "p_b")
    p_ab = _check_prob(p_ab, "p_ab")
    if p_ab > min(p_a, p_b) + 1e-12:
        raise ValueError("P(A and B) cannot exceed min(P(A), P(B))")
    return p_a + p_b - p_ab


def classify_events(p_a, p_b, p_ab, tol=1e-12):
    """Independent iff P(A and B) = P(A)P(B); exclusive iff P(A and B) = 0 -- Sec 2.2.3."""
    p_a = _check_prob(p_a, "p_a")
    p_b = _check_prob(p_b, "p_b")
    p_ab = _check_prob(p_ab, "p_ab")
    independent = abs(p_ab - p_a * p_b) <= tol
    exclusive = p_ab <= tol
    return independent, exclusive


def total_probability(priors, likelihoods):
    """P(Z) = sum_i P(Z|Ai) P(Ai) -- eqs (2.29), (2.55), (2.86)."""
    priors = _check_prob_vec(priors, "priors")
    likelihoods = _check_prob_vec(likelihoods, "likelihoods")
    if priors.shape != likelihoods.shape:
        raise ValueError("priors and likelihoods must have the same length")
    if abs(float(np.sum(priors)) - 1.0) > 1e-9:
        raise ValueError("priors must sum to 1 (complete, mutually exclusive set)")
    return float(np.sum(priors * likelihoods))


def bayes_simple(p_z_given_a, p_a, p_z):
    """P(A|Z) = P(Z|A) P(A) / P(Z) -- eq (2.51)."""
    p_z = _check_prob(p_z, "p_z")
    if p_z == 0.0:
        raise ValueError("P(Z) must be positive")
    return _check_prob(p_z_given_a, "p_z_given_a") * _check_prob(p_a, "p_a") / p_z


def bayes_explicit(p_a, p_z_given_a, p_z_given_not_a):
    """P(A|Z) with denominator expanded over A and not-A -- eq (2.52)."""
    p_a = _check_prob(p_a, "p_a")
    num = _check_prob(p_z_given_a, "p_z_given_a") * p_a
    den = num + _check_prob(p_z_given_not_a, "p_z_given_not_a") * (1.0 - p_a)
    if den == 0.0:
        raise ValueError("P(Z) = 0: event Z impossible under both branches")
    return num / den


def bayes_general(priors, likelihoods):
    """Posterior vector P(Ak|Z) -- eqs (2.53), (2.74)."""
    priors = _check_prob_vec(priors, "priors")
    likelihoods = _check_prob_vec(likelihoods, "likelihoods")
    p_z = total_probability(priors, likelihoods)
    if p_z == 0.0:
        raise ValueError("P(Z) = 0: no hypothesis can produce Z")
    return priors * likelihoods / p_z, p_z


def conditional_from_joint(p_a_and_b, p_a):
    """P(B|A) = P(A and B) / P(A) -- eq (2.48)."""
    p_a = _check_prob(p_a, "p_a")
    p_ab = _check_prob(p_a_and_b, "p_a_and_b")
    if p_a == 0.0:
        raise ValueError("P(A) must be positive")
    if p_ab > p_a + 1e-12:
        raise ValueError("P(A and B) cannot exceed P(A)")
    return p_ab / p_a


def conditional_subset(p_b, p_a):
    """P(B|A) = P(B)/P(A) when B is a subset of A -- eq (2.49)."""
    p_a = _check_prob(p_a, "p_a")
    p_b = _check_prob(p_b, "p_b")
    if p_a == 0.0:
        raise ValueError("P(A) must be positive")
    if p_b > p_a + 1e-12:
        raise ValueError("subset case needs P(B) <= P(A)")
    return p_b / p_a


def inclusion_exclusion_3(p_a, p_b, p_c, p_ab, p_ac, p_bc, p_abc):
    """P(A or B or C) -- eq (2.92)."""
    for v, nm in [(p_a, "p_a"), (p_b, "p_b"), (p_c, "p_c"), (p_ab, "p_ab"),
                  (p_ac, "p_ac"), (p_bc, "p_bc"), (p_abc, "p_abc")]:
        _check_prob(v, nm)
    return p_a + p_b + p_c - p_ab - p_ac - p_bc + p_abc


def at_least_one_of_iid(p, k):
    """P(at least one of k independent events, each prob p), inclusion-exclusion form.

    Equals 1 - (1-p)^k; the k = 3 expansion 3p - 3p^2 + p^3 is the book's
    dice example, eqs (2.93)-(2.96).
    """
    p = _check_prob(p, "p")
    k = _check_nonneg_int(k, "k")
    total = 0.0
    for j in range(1, k + 1):
        total += (-1) ** (j + 1) * math.comb(k, j) * p ** j
    closed = 1.0 - (1.0 - p) ** k
    if abs(total - closed) > 1e-12:
        raise AssertionError("inclusion-exclusion and complement forms disagree")
    return total


def exact_half_heads(n):
    """P(exactly n Heads in 2n fair flips) = C(2n, n) / 4^n -- eq (2.65)."""
    n = _check_nonneg_int(n, "n")
    return math.comb(2 * n, n) / 4.0 ** n


def stirling_factorial(n):
    """Stirling's approximation n! ~ n^n e^-n sqrt(2 pi n) -- eq (2.64)."""
    n = float(n)
    if n <= 0:
        raise ValueError("n must be positive")
    return n ** n * math.exp(-n) * math.sqrt(2.0 * math.pi * n)


def stirling_half_heads(n):
    """Stirling approximation P(n Heads in 2n flips) ~ 1/sqrt(pi n) -- eq (2.66)."""
    n = _check_nonneg_int(n, "n")
    if n == 0:
        raise ValueError("n must be >= 1")
    return 1.0 / math.sqrt(math.pi * n)


def suit_full_house_probability(n_suits=4, n_ranks=13, k_major=3, k_minor=2):
    """P(5-card hand has k_major cards of one suit, k_minor of another) -- eq (2.41)."""
    n_cards = n_suits * n_ranks
    hand = k_major + k_minor
    favorable = (n_suits * math.comb(n_ranks, k_major)
                 * (n_suits - 1) * math.comb(n_ranks, k_minor))
    total = math.comb(n_cards, hand)
    return favorable, total, favorable / total


def at_most_two_suits_probability(n_suits=4, n_ranks=13, hand=5):
    """P(5-card hand uses at most two suits) -- eqs (2.42)-(2.43).

    C(n_suits, 2) C(2 n_ranks, hand) counts every two-suit set; each
    single-suit hand is counted (n_suits - 1) times, so subtract the
    overcount (n_suits - 2) * n_suits * C(n_ranks, hand).
    """
    n_cards = n_suits * n_ranks
    pairs = math.comb(n_suits, 2) * math.comb(2 * n_ranks, hand)
    overcount = (n_suits - 2) * n_suits * math.comb(n_ranks, hand)
    favorable = pairs - overcount
    total = math.comb(n_cards, hand)
    return favorable, total, favorable / total


# ---------------------------------------------------------------- chapter 3

def _check_pmf(values, probs):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    probs = _check_prob_vec(probs, "probs")
    if values.shape != probs.shape:
        raise ValueError("values and probs must have the same length")
    if abs(float(np.sum(probs)) - 1.0) > 1e-9:
        raise ValueError("probabilities must sum to 1")
    return values, probs


def pmf_expectation(values, probs):
    """E(X) = sum x P(x) -- eq (3.4)."""
    values, probs = _check_pmf(values, probs)
    return float(np.sum(values * probs))


def pmf_variance(values, probs):
    """Var(X) = E[(X - mu)^2], with the computational form as cross-check -- eqs (3.19), (3.34)."""
    values, probs = _check_pmf(values, probs)
    mu = float(np.sum(values * probs))
    definition = float(np.sum(probs * (values - mu) ** 2))
    computational = float(np.sum(probs * values ** 2)) - mu ** 2
    if abs(definition - computational) > 1e-9 * max(1.0, abs(definition)):
        raise AssertionError("definition and computational forms disagree")
    return definition, mu


def joint_independent(joint, tol=1e-9):
    """X, Y independent iff joint pmf factorizes into its marginals -- eq (3.9)."""
    joint = np.asarray(joint, dtype=float)
    if joint.ndim != 2 or np.any(joint < 0):
        raise ValueError("joint must be a non-negative 2-D pmf table")
    if abs(float(joint.sum()) - 1.0) > 1e-9:
        raise ValueError("joint pmf must sum to 1")
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    return bool(np.max(np.abs(joint - np.outer(px, py))) <= tol), px, py


def pmf_sum_convolution(values_x, probs_x, values_y, probs_y):
    """pmf of X + Y for independent discrete X, Y -- Sec 3.1 (eq (3.11) example)."""
    vx, px = _check_pmf(values_x, probs_x)
    vy, py = _check_pmf(values_y, probs_y)
    acc = {}
    for a, pa in zip(vx, px):
        for b, pb in zip(vy, py):
            acc[a + b] = acc.get(a + b, 0.0) + pa * pb
    values = np.array(sorted(acc))
    probs = np.array([acc[v] for v in values])
    return values, probs


def expectation_linear(a, e_x, b, e_y, c):
    """E(aX + bY + c) = a E(X) + b E(Y) + c -- eq (3.13)."""
    return float(a) * float(e_x) + float(b) * float(e_y) + float(c)


def var_scale(a, var_x):
    """Var(aX) = a^2 Var(X) -- eq (3.24)."""
    var_x = float(var_x)
    if var_x < 0:
        raise ValueError("variance must be >= 0")
    return float(a) ** 2 * var_x


def var_sum_independent(variances):
    """Var(X1 + ... + Xn) = sum Var(Xi) for independent Xi -- eqs (3.25), (3.30)."""
    v = np.atleast_1d(np.asarray(variances, dtype=float))
    if np.any(v < 0):
        raise ValueError("variances must be >= 0")
    return float(np.sum(v))


def var_sum_with_cov(var_x, var_y, cov_xy):
    """Var(X + Y) = Var(X) + Var(Y) + 2 Cov(X, Y) -- eq (3.26) expansion."""
    var_x, var_y = float(var_x), float(var_y)
    if var_x < 0 or var_y < 0:
        raise ValueError("variances must be >= 0")
    cov_xy = float(cov_xy)
    if abs(cov_xy) > math.sqrt(var_x * var_y) + 1e-12:
        raise ValueError("|Cov| cannot exceed sqrt(VarX VarY)")
    return var_x + var_y + 2.0 * cov_xy


def bernoulli_variance(p):
    """Var(Bernoulli) = p(1-p) = pq -- eq (3.22)."""
    p = _check_prob(p, "p")
    return p * (1.0 - p)


def binomial_variance(n, p):
    """Var(#Heads in n biased flips) = npq -- eq (3.33)."""
    n = _check_nonneg_int(n, "n")
    return n * bernoulli_variance(p)


def sd_scale(a, sigma):
    """sigma_aX = |a| sigma -- eq (3.41)."""
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return abs(float(a)) * sigma


def sd_sum_independent(sigmas):
    """sigma of a sum of independent variables = sqrt(sum sigma_i^2) -- eqs (3.42)-(3.44)."""
    s = np.atleast_1d(np.asarray(sigmas, dtype=float))
    if np.any(s < 0):
        raise ValueError("sigmas must be >= 0")
    return float(math.sqrt(np.sum(s ** 2)))


def sd_bernoulli(p):
    """sigma = sqrt(pq) for one biased flip -- eq (3.46)."""
    return math.sqrt(bernoulli_variance(p))


def sd_binomial(n, p):
    """sigma = sqrt(npq) for n biased flips -- eq (3.47)."""
    return math.sqrt(binomial_variance(n, p))


def sd_fair_coin_sum(n):
    """sigma of #Heads in n fair flips = sqrt(n)/2 -- eqs (3.48), (3.51)."""
    n = _check_nonneg_int(n, "n")
    return math.sqrt(n) / 2.0


def sd_fair_coin_avg(n):
    """sigma of the average Heads fraction in n fair flips = 1/(2 sqrt(n)) -- eq (3.52)."""
    n = _check_nonneg_int(n, "n")
    if n == 0:
        raise ValueError("n must be >= 1")
    return 1.0 / (2.0 * math.sqrt(n))


def sd_of_mean(sigma, n):
    """Standard deviation of the mean: sigma / sqrt(n) -- eq (3.53)."""
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    n = _check_nonneg_int(n, "n")
    if n == 0:
        raise ValueError("n must be >= 1")
    return sigma / math.sqrt(n)


def sample_mean(x):
    """X-bar = (X1 + ... + Xn)/n -- eq (3.54)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.size == 0:
        raise ValueError("x must be non-empty")
    return float(np.mean(x))


def sd_of_mean_hetero(sigmas):
    """sigma of the average of n different variables = sqrt(sum sigma_i^2)/n -- eq (3.55)."""
    s = np.atleast_1d(np.asarray(sigmas, dtype=float))
    if np.any(s < 0) or s.size == 0:
        raise ValueError("sigmas must be a non-empty vector of >= 0 values")
    return float(math.sqrt(np.sum(s ** 2)) / s.size)


def population_variance(x):
    """s-tilde^2 = (1/n) sum (xi - xbar)^2, with computational cross-check -- eqs (3.37), (3.60), (3.66)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.size == 0:
        raise ValueError("x must be non-empty")
    xbar = float(np.mean(x))
    definition = float(np.mean((x - xbar) ** 2))
    computational = float(np.mean(x ** 2)) - xbar ** 2
    if abs(definition - computational) > 1e-9 * max(1.0, abs(definition)):
        raise AssertionError("definition and computational forms disagree")
    return definition


def sample_variance(x):
    """Unbiased sample variance s^2 = (1/(n-1)) sum (xi - xbar)^2 -- eq (3.73)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.size < 2:
        raise ValueError("sample variance needs n >= 2")
    xbar = float(np.mean(x))
    return float(np.sum((x - xbar) ** 2) / (x.size - 1))


def e_x_squared(sigma, mu):
    """E[X^2] = sigma^2 + mu^2 -- eq (3.70)."""
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return sigma ** 2 + float(mu) ** 2


def var_of_sample_mean(sigma, N):
    """E[(xbar - mu)^2] = sigma^2 / N -- eq (3.92); its sqrt is <= sigma -- eq (3.93)."""
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    N = _check_nonneg_int(N, "N")
    if N == 0:
        raise ValueError("N must be >= 1")
    return sigma ** 2 / N


# ---------------------------------------------------------------- chapter 4

def density_interval_probability(grid, density, a, b):
    """P(a <= X <= b) = integral of rho over [a, b], trapezoid on a grid -- eqs (4.2), (4.4)."""
    grid = np.atleast_1d(np.asarray(grid, dtype=float))
    density = np.atleast_1d(np.asarray(density, dtype=float))
    if grid.shape != density.shape or grid.size < 2:
        raise ValueError("grid and density must be equal-length vectors, n >= 2")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing")
    if np.any(density < 0):
        raise ValueError("density must be >= 0")
    total = float(np.trapezoid(density, grid))
    if abs(total - 1.0) > 0.05:
        raise ValueError(f"density integrates to {total:.4f}, not ~1")
    a, b = float(a), float(b)
    if not (grid[0] <= a <= b <= grid[-1]):
        raise ValueError("need grid[0] <= a <= b <= grid[-1]")
    xs = np.linspace(a, b, 513)
    ys = np.interp(xs, grid, density)
    return float(np.trapezoid(ys, xs))


def density_expectation(grid, density):
    """E(X) = integral of x rho(x) dx on a grid -- eqs (4.54)-(4.55)."""
    grid = np.atleast_1d(np.asarray(grid, dtype=float))
    density = np.atleast_1d(np.asarray(density, dtype=float))
    if grid.shape != density.shape or grid.size < 2:
        raise ValueError("grid and density must be equal-length vectors, n >= 2")
    return float(np.trapezoid(grid * density, grid))


def binomial_pmf(k, n, p):
    """P(k) = C(n,k) p^k (1-p)^(n-k) -- eqs (4.6), (4.8), (4.32), (4.60)."""
    n = _check_nonneg_int(n, "n")
    k = _check_nonneg_int(k, "k")
    p = _check_prob(p, "p")
    if k > n:
        return 0.0
    if p == 0.0:
        return float(k == 0)
    if p == 1.0:
        return float(k == n)
    if n <= 1000:
        return math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)
    log_pmf = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
               + k * math.log(p) + (n - k) * math.log1p(-p))
    return math.exp(log_pmf)


def binomial_pmf_vector(n, p):
    """Full binomial pmf over k = 0..n; sums to 1 -- eq (4.10)."""
    n = _check_nonneg_int(n, "n")
    p = _check_prob(p, "p")
    pmf = np.array([binomial_pmf(k, n, p) for k in range(n + 1)])
    if abs(float(pmf.sum()) - 1.0) > 1e-9:
        raise AssertionError("binomial pmf does not sum to 1")
    return pmf


def p_zero_equals_one(n):
    """p solving P(0) = P(1) for the binomial: p = 1/(n+1) -- eq (4.9)."""
    n = _check_nonneg_int(n, "n")
    if n == 0:
        raise ValueError("n must be >= 1")
    return 1.0 / (n + 1.0)


def binomial_mean(n, p):
    """E(k) = np -- eq (4.61) block."""
    return _check_nonneg_int(n, "n") * _check_prob(p, "p")


def binomial_second_moment(n, p):
    """E(k^2) = p^2 n(n-1) + pn -- eq (4.66)."""
    n = _check_nonneg_int(n, "n")
    p = _check_prob(p, "p")
    return p ** 2 * n * (n - 1) + p * n


def poisson_pmf(k, a):
    """P(k) = a^k e^-a / k! -- eq (4.40)."""
    k = _check_nonneg_int(k, "k")
    a = float(a)
    if a < 0:
        raise ValueError("a must be >= 0")
    return math.exp(k * math.log(a) - a - math.lgamma(k + 1)) if a > 0 else float(k == 0)


def poisson_small_interval(lam, eps):
    """P(1 event in a tiny interval) ~ lambda * eps -- eq (4.18)."""
    lam, eps = float(lam), float(eps)
    if lam < 0 or eps < 0:
        raise ValueError("lambda and eps must be >= 0")
    exact = poisson_pmf(1, lam * eps)
    return lam * eps, exact


def poisson_mean_rate(lam, t):
    """Expected number of events in time t: lambda t = sum k P_t(k) -- eq (4.19)."""
    lam, t = float(lam), float(t)
    if lam < 0 or t < 0:
        raise ValueError("lambda and t must be >= 0")
    a = lam * t
    ks = np.arange(0, max(20, int(a + 12 * math.sqrt(a + 1))))
    series = float(np.sum(ks * np.array([poisson_pmf(int(k), a) for k in ks])))
    if abs(series - a) > 1e-6 * max(1.0, a):
        raise AssertionError("series mean disagrees with lambda t")
    return a


def exponential_waiting_density(t, lam):
    """rho(t) = lambda e^(-lambda t): waiting time to the next event -- eqs (4.25)-(4.26)."""
    t, lam = float(t), float(lam)
    if t < 0 or lam <= 0:
        raise ValueError("need t >= 0 and lambda > 0")
    return lam * math.exp(-lam * t)


def exponential_interval_probability(t, dt, lam):
    """P(next event in [t, t+dt]) ~ e^(-lambda t) lambda dt -- eqs (4.23), (4.25)."""
    t, dt, lam = float(t), float(dt), float(lam)
    if t < 0 or dt < 0 or lam <= 0:
        raise ValueError("need t >= 0, dt >= 0, lambda > 0")
    return math.exp(-lam * t) * lam * dt


def exponential_crossing_time(rate_fast, rate_slow, ratio):
    """t solving e^(-r_fast t) = (1/ratio) e^(-r_slow t) -- eq (4.30) worked example."""
    rate_fast, rate_slow, ratio = float(rate_fast), float(rate_slow), float(ratio)
    if rate_fast <= rate_slow:
        raise ValueError("need rate_fast > rate_slow")
    if ratio <= 0:
        raise ValueError("ratio must be > 0")
    return math.log(ratio) / (rate_fast - rate_slow)


def exponential_moments(tau):
    """Mean tau, E(T^2) = 2 tau^2, Var = tau^2, with numeric cross-check -- eqs (4.83)-(4.86)."""
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be > 0")
    t = np.linspace(0.0, 60.0 * tau, 200001)
    rho = np.exp(-t / tau) / tau
    mean_num = float(np.trapezoid(t * rho, t))
    second_num = float(np.trapezoid(t ** 2 * rho, t))
    if abs(mean_num - tau) > 1e-4 * tau or abs(second_num - 2 * tau ** 2) > 1e-3 * tau ** 2:
        raise AssertionError("numeric integrals disagree with analytic moments")
    return tau, 2.0 * tau ** 2, tau ** 2


def binomial_poisson_limit(k, n, lam_eps_total):
    """Binomial P(k) with p = a/n against its Poisson limit -- eqs (4.34)-(4.37)."""
    a = float(lam_eps_total)
    n = _check_nonneg_int(n, "n")
    if n == 0:
        raise ValueError("n must be >= 1")
    exact = binomial_pmf(k, n, a / n)
    limit = poisson_pmf(k, a)
    return exact, limit, abs(exact - limit)


def poisson_zero_series(a, terms=60):
    """P(0) = e^-a as the alternating series 1 - a + a^2/2! - ... -- eq (4.53)."""
    a = float(a)
    if a < 0:
        raise ValueError("a must be >= 0")
    terms = _check_nonneg_int(terms, "terms")
    partial = 0.0
    partials = []
    for j in range(terms):
        partial += (-1) ** j * a ** j / math.factorial(j)
        partials.append(partial)
    closed = math.exp(-a)
    return partials, closed


def poisson_mode(a):
    """P(k) is maximized at k = ceil(a) - 1 (ties at a integer) -- eq (4.89)."""
    a = float(a)
    if a <= 0:
        raise ValueError("a must be > 0")
    k_star = int(math.ceil(a)) - 1
    return max(k_star, 0)


def poisson_mean_var(a, kmax=None):
    """Series check: sum k P(k) = a and sum k^2 P(k) - a^2 = a -- eqs (4.92)-(4.94)."""
    a = float(a)
    if a < 0:
        raise ValueError("a must be >= 0")
    if kmax is None:
        kmax = max(50, int(a + 15 * math.sqrt(a + 1)))
    ks = np.arange(0, kmax)
    pmf = np.array([poisson_pmf(int(k), a) for k in ks])
    mean = float(np.sum(ks * pmf))
    var = float(np.sum(ks ** 2 * pmf)) - mean ** 2
    return mean, var


def hypergeometric_pmf(k, N, K, n):
    """P(k) = C(K,k) C(N-K, n-k) / C(N,n) -- eq (4.71)."""
    N = _check_nonneg_int(N, "N")
    K = _check_nonneg_int(K, "K")
    n = _check_nonneg_int(n, "n")
    k = _check_nonneg_int(k, "k")
    if K > N or n > N:
        raise ValueError("need K <= N and n <= N")
    if k > min(K, n) or n - k > N - K:
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def hypergeometric_binomial_limit(k, n, p, N):
    """Hypergeometric -> binomial as N -> infinity at fixed p = K/N -- eqs (4.73), (4.75)."""
    K = int(round(float(p) * N))
    hyper = hypergeometric_pmf(k, N, K, n)
    binom_p = binomial_pmf(k, n, K / N)
    return hyper, binom_p, abs(hyper - binom_p)


def poisson_binomial_peak_ratio(n, p):
    """PP(pn)/PB(pn) = sqrt(1 - p) in the large-n Stirling limit -- eqs (4.95)-(4.98)."""
    n = _check_nonneg_int(n, "n")
    p = _check_prob(p, "p")
    k = int(round(p * n))
    if k == 0 or k == n:
        raise ValueError("pn must be an interior integer; increase n")
    pp = poisson_pmf(k, p * n)
    pb = binomial_pmf(k, n, p)
    if pb == 0.0:
        raise ValueError("binomial peak underflowed")
    return pp / pb, math.sqrt(1.0 - p)


# ---------------------------------------------------------------- chapter 5

def binomial_centered_pmf(x, n):
    """PB(x) = C(2n, n+x)/2^(2n): probability of n+x Heads in 2n flips -- eqs (5.3), (5.5)."""
    n = _check_nonneg_int(n, "n")
    x = int(x)
    if abs(x) > n:
        return 0.0
    return binomial_pmf(n + x, 2 * n, 0.5)


def gaussian_approx_2n(x, n):
    """PG(x) = e^(-x^2/n)/sqrt(pi n) for 2n fair flips -- eqs (5.4), (5.13)."""
    n = _check_nonneg_int(n, "n")
    if n == 0:
        raise ValueError("n must be >= 1")
    x = float(x)
    return math.exp(-x * x / n) / math.sqrt(math.pi * n)


def gaussian_approx_n(x, n):
    """PG(x) = e^(-2x^2/n)/sqrt(pi n/2) for n fair flips -- eq (5.14)."""
    n = _check_nonneg_int(n, "n")
    if n == 0:
        raise ValueError("n must be >= 1")
    x = float(x)
    return math.exp(-2.0 * x * x / n) / math.sqrt(math.pi * n / 2.0)


def gaussian_approx_biased(x, n, p):
    """PG(x) = e^(-x^2/(2npq))/sqrt(2 pi npq) for n biased flips -- eq (5.15)."""
    n = _check_nonneg_int(n, "n")
    p = _check_prob(p, "p")
    npq = n * p * (1.0 - p)
    if npq == 0:
        raise ValueError("npq must be > 0")
    x = float(x)
    return math.exp(-x * x / (2.0 * npq)) / math.sqrt(2.0 * math.pi * npq)


def poisson_stirling(k, a):
    """Poisson pmf with Stirling's formula applied to k! -- eqs (5.16)-(5.17)."""
    k = _check_nonneg_int(k, "k")
    if k == 0:
        raise ValueError("Stirling form needs k >= 1")
    a = float(a)
    if a <= 0:
        raise ValueError("a must be > 0")
    log_val = (k * math.log(a) - a
               - (k * math.log(k) - k + 0.5 * math.log(2.0 * math.pi * k)))
    return math.exp(log_val)


def poisson_gaussian(k, a):
    """PG(k) = e^(-(k-a)^2/(2a))/sqrt(2 pi a) -- eq (5.23)."""
    a = float(a)
    if a <= 0:
        raise ValueError("a must be > 0")
    k = float(k)
    return math.exp(-(k - a) ** 2 / (2.0 * a)) / math.sqrt(2.0 * math.pi * a)


def normal_pdf(x, mu, sigma):
    """Gaussian density f(x) = e^(-(x-mu)^2/(2 sigma^2))/sqrt(2 pi sigma^2) -- eqs (5.25), (5.28)."""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    x, mu = float(x), float(mu)
    return math.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2)) / math.sqrt(2.0 * math.pi * sigma ** 2)


def pmf_sd(values, probs):
    """sigma = sqrt(sum p (x - mu)^2) for a pmf -- eq (5.31), the sqrt lost by OCR."""
    variance, mu = pmf_variance(values, probs)
    return math.sqrt(variance), mu


# ---------------------------------------------------------------- chapter 6

def linear_model_stats(m, mu_x, sigma_x, mu_z, sigma_z):
    """Y = mX + Z with independent noise Z -- eqs (6.3)-(6.6), (6.17), (6.52), (6.76).

    Returns (mu_y, sigma_y, r) with mu_y = m mu_x + mu_z,
    sigma_y = sqrt(m^2 sigma_x^2 + sigma_z^2), r = m sigma_x / sigma_y.
    """
    m = float(m)
    sigma_x, sigma_z = float(sigma_x), float(sigma_z)
    if sigma_x < 0 or sigma_z < 0:
        raise ValueError("sigmas must be >= 0")
    mu_y = m * float(mu_x) + float(mu_z)
    sigma_y = math.sqrt(m ** 2 * sigma_x ** 2 + sigma_z ** 2)
    if sigma_y == 0:
        raise ValueError("degenerate model: sigma_y = 0")
    r = m * sigma_x / sigma_y
    return mu_y, sigma_y, r


def sample_cov(x, y):
    """Cov(x, y) = (1/n) sum (xi - xbar)(yi - ybar) -- eqs (6.10)-(6.12)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape or x.size < 2:
        raise ValueError("x and y must be equal-length vectors, n >= 2")
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def sample_r(x, y):
    """Sample correlation r = Cov(x,y)/(s_x s_y) -- eqs (6.12), (6.55)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    cov = sample_cov(x, y)
    sx = math.sqrt(float(np.mean((x - x.mean()) ** 2)))
    sy = math.sqrt(float(np.mean((y - y.mean()) ** 2)))
    if sx == 0 or sy == 0:
        raise ValueError("degenerate data: zero variance")
    return cov / (sx * sy)


def cov_shortcut(x, y):
    """Cov = mean(xy) - mean(x) mean(y) -- eqs (6.8), (6.14), (6.63)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape or x.size < 2:
        raise ValueError("x and y must be equal-length vectors, n >= 2")
    direct = sample_cov(x, y)
    shortcut = float(np.mean(x * y)) - float(np.mean(x)) * float(np.mean(y))
    if abs(direct - shortcut) > 1e-9 * max(1.0, abs(direct)):
        raise AssertionError("covariance forms disagree")
    return shortcut


def slope_from_cov(x, y):
    """Model slope m = Cov(x,y)/s_x^2 -- eq (6.13)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    var_x = float(np.mean((x - x.mean()) ** 2))
    if var_x == 0:
        raise ValueError("zero variance in x")
    return sample_cov(x, y) / var_x


def best_constant_predictor(y):
    """The constant prediction minimizing E[(Y - yp)^2] is the mean -- eqs (6.22)-(6.23)."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if y.size == 0:
        raise ValueError("y must be non-empty")
    mu = float(np.mean(y))
    mse_mu = float(np.mean((y - mu) ** 2))
    for cand in (mu - 0.1 * (1 + abs(mu)), mu + 0.1 * (1 + abs(mu))):
        if float(np.mean((y - cand) ** 2)) < mse_mu:
            raise AssertionError("a constant beat the mean; impossible")
    return mu, mse_mu


def prediction_improvement(r):
    """Fractional reduction in mean squared prediction error: 1 - r^2 -- eq (6.27)."""
    r = float(r)
    if not -1.0 <= r <= 1.0:
        raise ValueError("r must be in [-1, 1]")
    return 1.0 - r * r


def reverse_regression_slope(r, sigma_x, sigma_y):
    """X predicted from Y: slope r sigma_x / sigma_y -- eqs (6.36), (6.74)."""
    r = float(r)
    sigma_x, sigma_y = float(sigma_x), float(sigma_y)
    if not -1.0 <= r <= 1.0:
        raise ValueError("r must be in [-1, 1]")
    if sigma_y <= 0 or sigma_x < 0:
        raise ValueError("sigmas must be positive")
    return r * sigma_x / sigma_y


def regression_to_mean_factor(r):
    """Second-test group average: yavg = r^2 y1 -- eqs (6.39)-(6.40)."""
    r = float(r)
    if not -1.0 <= r <= 1.0:
        raise ValueError("r must be in [-1, 1]")
    return r * r


def least_squares_fit(x, y):
    """Least-squares line y = Ax + B -- eqs (6.42)-(6.49), (6.82), (6.92).

    A = (<xy> - <x><y>)/(<x^2> - <x>^2), B = <y> - A<x>; residuals sum to 0.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape or x.size < 2:
        raise ValueError("x and y must be equal-length vectors, n >= 2")
    mx, my = float(np.mean(x)), float(np.mean(y))
    mxy, mx2 = float(np.mean(x * y)), float(np.mean(x * x))
    denom = mx2 - mx * mx
    if denom == 0:
        raise ValueError("all x identical: slope undefined")
    A = (mxy - mx * my) / denom
    B_first = (my * mx2 - mx * mxy) / denom
    B_second = my - A * mx
    if abs(B_first - B_second) > 1e-9 * max(1.0, abs(B_first)):
        raise AssertionError("the two intercept forms disagree")
    resid = y - (A * x + B_second)
    S = float(np.sum(resid ** 2))
    if abs(float(np.sum(resid))) > 1e-9 * max(1.0, float(np.sum(np.abs(resid)))):
        raise AssertionError("residuals do not sum to zero")
    return A, B_second, S


def regression_slope_product(x, y):
    """Slopes of y-on-x and x-on-y multiply to r^2 -- Sec 6.7 (with eq (6.55))."""
    A, _, _ = least_squares_fit(x, y)
    C, _, _ = least_squares_fit(y, x)
    r = sample_r(x, y)
    if abs(A * C - r * r) > 1e-9 * max(1.0, r * r):
        raise AssertionError("A*C != r^2")
    return A, C, r


def joint_density_factorizes(gx, dx, gy, dy):
    """Independent continuous variables: rho(x,y) = rho_x(x) rho_y(y) -- eqs (6.64)-(6.65)."""
    gx = np.atleast_1d(np.asarray(gx, dtype=float))
    dx = np.atleast_1d(np.asarray(dx, dtype=float))
    gy = np.atleast_1d(np.asarray(gy, dtype=float))
    dy = np.atleast_1d(np.asarray(dy, dtype=float))
    if gx.shape != dx.shape or gy.shape != dy.shape:
        raise ValueError("grid/density shape mismatch")
    joint = np.outer(dx, dy)
    total = float(np.trapezoid(np.trapezoid(joint, gy, axis=1), gx))
    return joint, total


def sum_density_convolution(gx, dx, gy, dy, z):
    """Density of Z = X + Y at z: integral rho_x(x) rho_y(z - x) dx -- eqs (6.65)-(6.66)."""
    gx = np.atleast_1d(np.asarray(gx, dtype=float))
    dx = np.atleast_1d(np.asarray(dx, dtype=float))
    gy = np.atleast_1d(np.asarray(gy, dtype=float))
    dy = np.atleast_1d(np.asarray(dy, dtype=float))
    z = float(z)
    vals = dx * np.interp(z - gx, gy, dy, left=0.0, right=0.0)
    return float(np.trapezoid(vals, gx))


def gaussian_sum_density(z, sigma_x, sigma_y):
    """Z = X + Y for independent zero-mean Gaussians: N(0, sx^2 + sy^2) -- eq (6.70)."""
    sigma_x, sigma_y = float(sigma_x), float(sigma_y)
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValueError("sigmas must be > 0")
    s = math.sqrt(sigma_x ** 2 + sigma_y ** 2)
    return normal_pdf(z, 0.0, s)


def excess_score_factor(r):
    """sigma_y(1-r) / (sigma_y sqrt(1-r^2)) = sqrt((1-r)/(1+r)) -- eq (6.81)."""
    r = float(r)
    if not -1.0 < r < 1.0:
        raise ValueError("r must be in (-1, 1)")
    ratio = (1.0 - r) / math.sqrt(1.0 - r * r)
    closed = math.sqrt((1.0 - r) / (1.0 + r))
    if abs(ratio - closed) > 1e-12:
        raise AssertionError("algebraic identity failed")
    return closed


# ---------------------------------------------------------------- chapter 7

def exp_taylor(x, terms=30):
    """e^x = sum x^k / k! -- Appendix eq (7.7); e^x ~ 1 + x for small x -- eq (7.9)."""
    x = float(x)
    terms = _check_nonneg_int(terms, "terms")
    partial = 0.0
    partials = []
    for k in range(terms):
        partial += x ** k / math.factorial(k)
        partials.append(partial)
    return partials, math.exp(x)


def one_plus_a_to_n(a, n, order=1):
    """(1+a)^n against the approximation ladder -- eqs (7.14), (7.21), (7.23)-(7.24).

    order 1: e^(na) (valid na^2 << 1); order 2: e^(na) e^(-na^2/2)
    (valid na^3 << 1).
    """
    a = float(a)
    n = float(n)
    if a <= -1.0:
        raise ValueError("need a > -1")
    exact = (1.0 + a) ** n
    if order == 1:
        approx = math.exp(n * a)
        validity = abs(n * a * a)
    elif order == 2:
        approx = math.exp(n * a - n * a * a / 2.0)
        validity = abs(n * a ** 3)
    else:
        raise ValueError("order must be 1 or 2")
    return exact, approx, validity


def power_derivative_quotient(x, n, delta):
    """((x+d)^n - x^n)/d = n x^(n-1) + O(d) via the binomial expansion -- eqs (7.31)-(7.35)."""
    x, delta = float(x), float(delta)
    n = _check_nonneg_int(n, "n")
    if delta == 0:
        raise ValueError("delta must be nonzero")
    quotient = ((x + delta) ** n - x ** n) / delta
    derivative = n * x ** (n - 1) if n >= 1 else 0.0
    return quotient, derivative
