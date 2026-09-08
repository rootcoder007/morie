"""Nonparametric Bayes predictive value of a new observation.

A quantile pyramid turns the usual nonparametric Bayes construction
inside out. A Polya tree fixes the partition of the sample space and
puts random mass on it; a quantile pyramid fixes the MASS -- a half, two
quarters, four eighths -- and lets the partition be random. The
arbitrariness of "which partition?" disappears, because the cut points
are the median, the quartiles, the octiles, which are the same objects
whatever the scale.

The construction is a recursion down the dyadic rationals. Writing
Q(y) for the quantile function, level m fills in the odd dyadics from
the level above it:

    Q_m(j/2^m) = Q_{m-1}((j-1)/2^m) (1 - V_{m,j})
                 + Q_{m-1}((j+1)/2^m) V_{m,j},   j = 1, 3, ..., 2^m - 1

so each new quantile lands somewhere strictly between the two quantiles
that bracket it, at a random point V of the way across. The Beta
quantile pyramid takes V ~ Beta(a_m/2, a_m/2), symmetric about a half,
and a_m growing with the level -- a_m = c m^3 is the paper's example --
which is what makes the limit absolutely continuous rather than a
singular mess. Centring the prior somewhere other than uniform is a
shift of the MEAN of V, and the paper gives that mean exactly: the
fraction of the parent interval that the prior guess assigns to the left
child.

Down at level m the quantile function is linear between the 2^m cut
points, so the density is a histogram with fixed cell probabilities 1/k
and random cell widths,

    f(x) = (1/k) / (q_j - q_{j-1})   for x in (q_{j-1}, q_j],  k = 2^m

and that is the likelihood. Two of them, in fact, and they are not the
same inference:

  "exact"       the honest likelihood of the model above, a product of
                the cell densities over the observations.
  "substitute"  the multinomial probability of the observed cell counts
                under equal cell probabilities. This is Jeffreys'
                substitution likelihood. It is NOT the conditional
                distribution of the data given any statistic, and it is
                known to be conservative -- which is a reason to offer
                it, not a reason to hide it.

What comes out is the thing the question actually asks for: the
predictive distribution of a new observation. Averaging the random
histogram over the posterior draws gives a predictive density, a
predictive distribution function, predictive quantiles, and the
predictive mean and standard deviation -- the last two computed from the
cellwise moments of the histogram rather than from the draws' locations,
so they are exact given the draws instead of being a second Monte Carlo
approximation stacked on the first.

References
  Hjort, N.L. and Walker, S.G. (2009) "Quantile pyramids for Bayesian
    nonparametrics." The Annals of Statistics 37(1), 105-131.
    doi:10.1214/07-AOS553. The construction (their equation 3), the
    Beta pyramid and the a_m = c m^3 schedule (section 4.1), the
    centring identity (equation 6), the random-histogram density
    (equation 9), the exact likelihood (equation 10), the multinomial
    substitute likelihood (equation 11), the factorised prior
    (equation 15) and both Metropolis-Hastings acceptance ratios
    (section 6).
  Jeffreys, H. (1967) "Theory of Probability," 3rd edition. Oxford
    University Press, chapter 4. The substitution likelihood for the
    median.
  Lavine, M. (1995) "On an approximate likelihood for quantiles."
    Biometrika 82(1), 220-222. Why the substitute likelihood gives
    conservative inference.
  Ferguson, T.S. (1974) "Prior distributions on spaces of probability
    measures." The Annals of Statistics 2(4), 615-629. The singular
    limit when the concentration does not grow with the level.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["bnppvl", "np_predictive_value", "pyramid_draw",
           "pyramid_log_prior", "cell_counts", "log_likelihood",
           "LIKELIHOODS", "CENTRINGS", "SCHEDULES", "INITS", "cheatsheet"]

LIKELIHOODS = ("exact", "substitute")
INITS = ("prior", "empirical")
CENTRINGS = ("uniform", "null")
SCHEDULES = ("cubic", "constant")


def _concentration(level, c, schedule):
    """The Beta concentration a_m at a given level.

    "cubic" is the paper's a_m = c m^3. The sum of 1/a_m^(1/2) then
    converges, which is the condition that buys an absolutely
    continuous limit. "constant" holds a_m fixed, which does NOT: the
    limit is continuous but singular, and it is here because a reader
    who wants to see that happen should be able to.
    """
    if schedule == "cubic":
        return float(c) * level * level * level
    if schedule == "constant":
        return float(c)
    raise ValueError("schedule must be one of %r" % (SCHEDULES,))


def _null_mean(nullq, j, level):
    """The mean of V that centres the pyramid on a prior guess.

    The fraction of the parent interval that the guess assigns to the
    left child -- the paper's equation (6). A guess that is flat over
    the parent gives exactly a half, which is the symmetric case.
    """
    d = 1.0 / (1 << level)
    a = nullq((j - 1) * d)
    b = nullq(j * d)
    cc = nullq((j + 1) * d)
    wide = cc - a
    if wide <= 0.0:
        raise ValueError("the centring quantile function is not strictly "
                         "increasing on the dyadic grid")
    return (b - a) / wide


def _ab(level, c, schedule, centring, nullq, j):
    """The Beta parameters for one node."""
    a_m = _concentration(level, c, schedule)
    if a_m <= 0.0:
        raise ValueError("the concentration must be positive")
    if centring == "uniform":
        return 0.5 * a_m, 0.5 * a_m
    mu = _null_mean(nullq, j, level)
    if not 0.0 < mu < 1.0:
        raise ValueError("the centring puts a node on the edge of its "
                         "parent interval")
    return a_m * mu, a_m * (1.0 - mu)


def pyramid_draw(rng, m, c=2.5, schedule="cubic", centring="uniform",
                 nullq=None):
    """One draw of the quantile pyramid down to level m.

    Returns the full dyadic grid of length 2^m + 1, with 0 and 1 at the
    ends. Levels are filled in order and, within a level, the odd nodes
    left to right, so two implementations that consume the same random
    stream in that order produce the same pyramid.
    """
    m = int(m)
    if m < 1:
        raise ValueError("the pyramid needs at least one level")
    k = 1 << m
    q = [0.0] * (k + 1)
    q[k] = 1.0
    for level in range(1, m + 1):
        step = 1 << (m - level)
        j = 1
        while j < (1 << level):
            i = j * step
            a, b = _ab(level, c, schedule, centring, nullq, j)
            v = float(rng.beta(a, b))
            q[i] = q[i - step] * (1.0 - v) + q[i + step] * v
            j += 2
    return q


def _log_beta_density(v, a, b):
    if not 0.0 < v < 1.0:
        return float("-inf")
    return ((a - 1.0) * math.log(v) + (b - 1.0) * math.log1p(-v)
            + _w.lgamma(a + b) - _w.lgamma(a) - _w.lgamma(b))


def pyramid_log_prior(q, m, c=2.5, schedule="cubic", centring="uniform",
                      nullq=None):
    """The log prior density of a pyramid, factorised level by level.

    Each node contributes the density of its own V plus the Jacobian of
    the map from V to the node's position, which is one over the width
    of the parent interval. That Jacobian is the piece it is easy to
    drop, and dropping it silently tilts every acceptance ratio.
    """
    m = int(m)
    terms = []
    for level in range(1, m + 1):
        step = 1 << (m - level)
        j = 1
        while j < (1 << level):
            i = j * step
            lo = q[i - step]
            hi = q[i + step]
            wide = hi - lo
            if wide <= 0.0 or not lo < q[i] < hi:
                return float("-inf")
            a, b = _ab(level, c, schedule, centring, nullq, j)
            terms.append(_log_beta_density((q[i] - lo) / wide, a, b)
                         - math.log(wide))
            j += 2
    return _w.csum(terms)


def _cell_of(u, q, k):
    """The index j in 1..k of the cell holding u, ties to the left cell.

    A linear scan, not a bisection: k is 2^m with m small by
    construction, and a scan visits the cells in one fixed order in
    both arms.
    """
    for j in range(1, k + 1):
        if u <= q[j]:
            return j
    return k


def cell_counts(u, q):
    """Counts of the observations falling in each of the k cells."""
    k = len(q) - 1
    n = [0] * (k + 1)
    for v in u:
        n[_cell_of(v, q, k)] += 1
    return n


def log_likelihood(u, q, kind="exact"):
    """The log likelihood of the cell configuration.

    "exact" is the product of the random-histogram densities. The
    "substitute" version is the multinomial probability of the counts
    under equal cell probabilities, which does not depend on the widths
    at all -- and that is precisely the property that makes it
    conservative.
    """
    if kind not in LIKELIHOODS:
        raise ValueError("kind must be one of %r" % (LIKELIHOODS,))
    k = len(q) - 1
    cnt = cell_counts(u, q)
    n = sum(cnt)
    if kind == "exact":
        terms = []
        for j in range(1, k + 1):
            if cnt[j]:
                w = q[j] - q[j - 1]
                if w <= 0.0:
                    return float("-inf")
                terms.append(cnt[j] * (-math.log(float(k)) - math.log(w)))
        return _w.csum(terms)
    terms = [_w.lgamma(n + 1.0), -n * math.log(float(k))]
    for j in range(1, k + 1):
        terms.append(-_w.lgamma(cnt[j] + 1.0))
    return _w.csum(terms)


def _predictive_pieces(draws, k):
    """Cellwise first and second moments of the random histogram.

    The mean of a cell is its midpoint and the second moment is
    (a^2 + ab + b^2)/3, because the histogram is uniform inside the
    cell. Taking these in closed form rather than from the sampled
    positions is what keeps the predictive moments exact given the
    draws.
    """
    m1 = []
    m2 = []
    for q in draws:
        t1 = []
        t2 = []
        for j in range(1, k + 1):
            a = q[j - 1]
            b = q[j]
            t1.append(0.5 * (a + b) / k)
            t2.append((a * a + a * b + b * b) / (3.0 * k))
        m1.append(_w.csum(t1))
        m2.append(_w.csum(t2))
    return m1, m2


def _density_at(u, q, k):
    j = _cell_of(u, q, k)
    w = q[j] - q[j - 1]
    return 0.0 if w <= 0.0 else 1.0 / (k * w)


def _cdf_at(u, q, k):
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    j = _cell_of(u, q, k)
    w = q[j] - q[j - 1]
    if w <= 0.0:
        return (j - 1.0) / k
    return (j - 1.0) / k + (u - q[j - 1]) / (k * w)


def _empirical_start(u, k):
    """A starting pyramid halfway between the data and the uniform one.

    The empirical quantiles alone will not do: with ties, or with fewer
    observations than cells, two of them can coincide and a quantile
    function with a zero-width cell is not a quantile function at all.
    Averaging with the uniform grid guarantees a strict increase of at
    least one over twice the cell count while still starting the chain
    where the data are, which is worth a great many sweeps of a
    single-site sampler.
    """
    n = len(u)
    su = sorted(u)
    q = [0.0] * (k + 1)
    q[k] = 1.0
    for j in range(1, k):
        idx = int(math.ceil(j * n / float(k))) - 1
        if idx < 0:
            idx = 0
        if idx > n - 1:
            idx = n - 1
        q[j] = 0.5 * (su[idx] + j / float(k))
    return q


def np_predictive_value(x, m=4, c=2.5, schedule="cubic",
                        centring="uniform", nullq=None, likelihood="exact",
                        lo=0.0, hi=1.0, sweeps=400, burn=100, thin=2,
                        seed=0, init="prior", grid=None,
                        probs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Fit a Beta quantile pyramid and report the predictive for a new draw.

    Parameters
    ----------
    x : sequence
        The observations. They must lie inside [lo, hi].
    m : int
        Pyramid depth. The histogram has k = 2^m cells.
    c : float
        Concentration scale. Larger holds the pyramid closer to its
        centring.
    schedule : str
        A member of SCHEDULES.
    centring : str
        A member of CENTRINGS. "null" centres on `nullq`.
    nullq : callable or None
        The centring quantile function on [0, 1], strictly increasing.
    likelihood : str
        A member of LIKELIHOODS.
    lo, hi : float
        The support. The data are mapped linearly onto the unit
        interval and everything is mapped back at the end.
    sweeps, burn, thin : int
        Metropolis-Hastings sweeps in total, sweeps discarded, and the
        keep-every rate. Each sweep proposes every interior quantile
        once, in order.
    seed : int
        The random stream.
    init : str
        A member of INITS. "prior" starts the chain at a draw from the
        prior, which is what the paper describes; "empirical" starts it
        near the data, which reaches the same posterior in far fewer
        sweeps of a single-site sampler.
    grid : sequence or None
        Points at which the predictive density and distribution
        function are reported.
    probs : sequence
        Predictive quantile levels.

    Returns
    -------
    RichResult
        The posterior mean quantile function, the predictive mean and
        standard deviation of a new observation, the predictive density
        and distribution function on the grid, the predictive
        quantiles, and the acceptance rate.

    References
    ----------
    Hjort and Walker (2009) Annals of Statistics 37(1), 105-131.
    """
    if likelihood not in LIKELIHOODS:
        raise ValueError("likelihood must be one of %r" % (LIKELIHOODS,))
    if centring not in CENTRINGS:
        raise ValueError("centring must be one of %r" % (CENTRINGS,))
    if schedule not in SCHEDULES:
        raise ValueError("schedule must be one of %r" % (SCHEDULES,))
    if init not in INITS:
        raise ValueError("init must be one of %r" % (INITS,))
    if centring == "null" and nullq is None:
        raise ValueError("centring on a null distribution needs nullq")
    m = int(m)
    if m < 1:
        raise ValueError("the pyramid needs at least one level")
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        raise ValueError("hi must exceed lo")
    xs = [float(v) for v in x]
    if not xs:
        raise ValueError("need at least one observation")
    if any(v < lo or v > hi for v in xs):
        raise ValueError("every observation must lie inside [lo, hi]")
    sweeps = int(sweeps)
    burn = int(burn)
    thin = int(thin)
    if sweeps < 1 or burn < 0 or burn >= sweeps or thin < 1:
        raise ValueError("need 0 <= burn < sweeps and thin >= 1")
    k = 1 << m
    u = [(v - lo) / (hi - lo) for v in xs]
    n = len(u)

    rng = _core._SplitMix64(seed)
    if init == "prior":
        q = pyramid_draw(rng, m, c, schedule, centring, nullq)
    else:
        q = _empirical_start(u, k)
    lp = pyramid_log_prior(q, m, c, schedule, centring, nullq)
    cnt = cell_counts(u, q)

    draws = []
    tried = 0
    taken = 0
    for it in range(sweeps):
        for j in range(1, k):
            a = q[j - 1]
            b = q[j + 1]
            prop = a + (b - a) * float(rng.uniform())
            tried += 1
            if not a < prop < b:
                # A proposal that lands exactly on a neighbour is not a
                # valid quantile function; reject rather than divide by
                # a zero width.
                rng.uniform()
                continue
            qq = list(q)
            qq[j] = prop
            lpp = pyramid_log_prior(qq, m, c, schedule, centring, nullq)
            # Only the two cells either side of j change, so only their
            # counts have to be recomputed.
            left = 0
            right = 0
            for v in u:
                # The leftmost cell is CLOSED at its lower end, exactly
                # as the cell lookup treats it. Writing a strict
                # inequality here loses every observation sitting on the
                # lower bound of the support, and the likelihood then
                # never squeezes the first cell.
                if (v > a or j == 1) and v <= prop:
                    left += 1
                elif prop < v <= b:
                    right += 1
            if likelihood == "exact":
                # The paper's ratio: the CURRENT widths raised to the
                # current counts over the proposed widths raised to the
                # proposed counts, because the density is one over the
                # width.
                num = (cnt[j] * math.log(q[j] - a)
                       + cnt[j + 1] * math.log(b - q[j]) + lpp)
                den = (left * math.log(prop - a)
                       + right * math.log(b - prop) + lp)
            else:
                num = (_w.lgamma(cnt[j] + 1.0) + _w.lgamma(cnt[j + 1] + 1.0)
                       + lpp)
                den = (_w.lgamma(left + 1.0) + _w.lgamma(right + 1.0) + lp)
            logr = num - den
            acc = float(rng.uniform())
            if logr >= 0.0 or (acc > 0.0 and math.log(acc) < logr):
                q = qq
                lp = lpp
                cnt[j] = left
                cnt[j + 1] = right
                taken += 1
        if it >= burn and (it - burn) % thin == 0:
            draws.append(list(q))
    if not draws:
        raise ValueError("no draws were kept; lower burn or thin")

    d = len(draws)
    m1, m2 = _predictive_pieces(draws, k)
    e1 = _w.csum(m1) / d
    e2 = _w.csum(m2) / d
    v1 = e2 - e1 * e1
    span = hi - lo
    pred_mean = lo + span * e1
    pred_sd = span * math.sqrt(v1) if v1 > 0.0 else 0.0

    qbar = []
    for i in range(k + 1):
        qbar.append(lo + span * (_w.csum(dr[i] for dr in draws) / d))

    if grid is None:
        grid = [lo + span * (t + 0.5) / 8.0 for t in range(8)]
    grid = [float(v) for v in grid]
    dens = []
    cdf = []
    for g in grid:
        gu = (g - lo) / span
        dens.append(_w.csum(_density_at(gu, dr, k) for dr in draws)
                    / (d * span))
        cdf.append(_w.csum(_cdf_at(gu, dr, k) for dr in draws) / d)

    def _F(t):
        return _w.csum(_cdf_at(t, dr, k) for dr in draws) / d

    pq = []
    for p in probs:
        p = float(p)
        if not 0.0 < p < 1.0:
            raise ValueError("every predictive probability must lie "
                             "strictly inside (0, 1)")
        # A fixed number of bisection steps on the averaged distribution
        # function: the predictive CDF is a mixture of piecewise linear
        # pieces and has no closed-form inverse, and a fixed step count
        # is the only root find that cannot iterate a different number
        # of times in the two arms.
        pq.append(lo + span * _w.bisect(lambda t: _F(t) - p, 0.0, 1.0))

    ll = log_likelihood(u, [(v - lo) / span for v in qbar], likelihood)
    return RichResult(payload={
        "quantile_mean": qbar,
        "estimate": pred_mean,
        "se": pred_sd / math.sqrt(d),
        "predictive_mean": pred_mean,
        "predictive_sd": pred_sd,
        "grid": grid,
        "density": dens,
        "cdf": cdf,
        "probs": [float(p) for p in probs],
        "predictive_quantile": pq,
        "counts": cnt[1:],
        "log_likelihood": ll,
        "log_prior": lp,
        "accept_rate": taken / float(tried) if tried else float("nan"),
        "n_draws": d,
        "n": n,
        "k": k,
        "m": m,
        "c": float(c),
        "lo": lo,
        "hi": hi,
        "schedule": schedule,
        "centring": centring,
        "likelihood": likelihood,
        "init": init,
        "method": "Beta quantile pyramid predictive",
    })


bnppvl = np_predictive_value


def cheatsheet():
    return ("bnppvl: quantile-pyramid predictive for a new observation. "
            "likelihoods " + ", ".join(LIKELIHOODS) + "; inits "
            + ", ".join(INITS) + "; centrings "
            + ", ".join(CENTRINGS) + "; schedules " + ", ".join(SCHEDULES))
