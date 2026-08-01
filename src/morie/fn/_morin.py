"""Shared primitives for the Morin probability shelf.

Every function here implements a named result from

    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace. Chapters 1-6.

The equation numbers in the per-module front ends refer to that book.
Counting primitives use exact integer arithmetic (math.factorial /
math.comb); probability helpers validate their inputs and fail loudly.
"""

import math

import numpy as np

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
