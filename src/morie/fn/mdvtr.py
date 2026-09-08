# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Median voter theorem (Black 1948)."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["median_voter", "mdvtr", "condorcet_winner"]

_METHOD = "Median voter theorem (Black 1948)"


def _binom_cdf(k, n):
    """P(X <= k) for X ~ Binomial(n, 1/2), by exact log-summation."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    lg = math.lgamma
    tot = 0.0
    for i in range(int(k) + 1):
        tot += math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1)
                        - n * math.log(2.0))
    return min(tot, 1.0)


def _order_statistic_ci(x_sorted, alpha):
    """Distribution-free interval for the median, from order statistics.

    The count of observations below the true median is Binomial(n, 1/2)
    whatever the underlying distribution is, so the coverage of
    ``[x_(k), x_(n+1-k)]`` depends on nothing but n. The cost is
    granularity: coverage moves in steps and cannot be set to exactly
    ``1 - alpha`` at small n, which is why the achieved level is
    returned alongside the interval rather than left implied.
    """
    n = x_sorted.size
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    k = 0
    for cand in range(1, n // 2 + 1):
        if _binom_cdf(cand - 1, n) <= alpha / 2:
            k = cand
        else:
            break
    if k < 1:
        return float("nan"), float("nan"), float("nan")
    cover = 1.0 - 2.0 * _binom_cdf(k - 1, n)
    return float(x_sorted[k - 1]), float(x_sorted[n - k]), float(cover)


def _kde_at(x, point):
    """Gaussian kernel density estimate at one point, Silverman's rule."""
    n = x.size
    s = float(np.std(x, ddof=1))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    spread = min(s, iqr / 1.349) if iqr > 0 else s
    if spread <= 0:
        return float("nan")
    h = 0.9 * spread * n ** (-0.2)
    u = (x - point) / h
    return float(np.mean(np.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)) / h)


def _z(q):
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def median_voter(x, alpha=0.05, alternatives=None):
    """The Condorcet winner on a single dimension is the median.

    With single-peaked preferences over one dimension, the median ideal
    point beats every alternative in a pairwise majority vote. It is the
    *median*, not the mean: the theorem is about counting voters, so
    moving one extremist further out shifts the mean and leaves the
    winner exactly where it was.

    Three things the usual statement glosses over, which the payload
    reports rather than assumes:

    **The electorate should be odd.** With an even number of voters
    every point in the closed interval between the two central ideal
    points is a Condorcet winner -- the winner is a set, not a point.
    ``median_interval`` gives that set and ``unique_winner`` says
    whether it collapses to one point.

    **The standard error usually quoted is not general.** The familiar
    :math:`1.2533\\,s/\\sqrt{n}` is
    :math:`\\sqrt{\\pi/2}\\,\\sigma/\\sqrt{n}`, which is the asymptotic
    standard error of the sample median *only when the data are
    normal*. The general result is :math:`1/(2 f(m)\\sqrt{n})`, and the
    two part company as soon as the distribution is not normal. For a
    heavy-tailed electorate the normal formula badly overstates the
    uncertainty, because it reads the tails as spread when the median
    responds only to the density at the centre. Both are returned,
    along with a distribution-free interval from the order statistics
    that assumes neither.

    **Single-peakedness is an assumption, not a property of the ideal
    points, and passing ``alternatives`` does not test it.** This is
    worth being blunt about, because the check reads as though it
    might. Preferences here are constructed from distance on the line:
    a voter is taken to prefer whichever platform lies nearer their
    ideal point. Euclidean preferences on one dimension are
    single-peaked *by construction*, so the median wins every pairwise
    contest as a matter of arithmetic and ``condorcet_verified`` can
    never come back False. It verifies the implementation, not the
    assumption -- hence ``check_is_definitional``.

    To test the assumption one needs preferences that were not built
    from distance. :func:`condorcet_winner` takes a utility matrix and
    finds the pairwise-majority winner directly, returning None when
    the majority relation cycles.

    Parameters
    ----------
    x : array-like
        Voter ideal points on a single dimension.
    alpha : float
        Two-sided level for the intervals.
    alternatives : array-like, optional
        Platforms to put against the median pairwise.
        ``condorcet_verified`` records whether the median beat or tied
        every one of them.

    Returns
    -------
    RichResult with ``estimate``, ``se`` (density-based), ``se_normal``,
    ``ci_lower``/``ci_upper``, ``ci_exact_lower``/``ci_exact_upper``,
    ``exact_coverage``, ``median_interval``, ``unique_winner``,
    ``condorcet_verified``, ``density_at_median``.

    References
    ----------
    Black D (1948) On the rationale of group decision-making,
    *Journal of Political Economy* 56(1):23-34.
    Downs A (1957) *An Economic Theory of Democracy*.

    Examples
    --------
    >>> r = median_voter([1.0, 2.0, 3.0, 4.0, 100.0])
    >>> r["estimate"]
    3.0
    >>> r["unique_winner"]
    True
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return RichResult(payload={"estimate": np.nan, "se": np.nan, "n": 0,
                                   "method": _METHOD})
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
    xs = np.sort(x)
    est = float(np.median(xs))
    if n % 2 == 1:
        interval = (est, est)
        unique = True
    else:
        interval = (float(xs[n // 2 - 1]), float(xs[n // 2]))
        unique = interval[0] == interval[1]

    if n > 1:
        se_normal = float(1.2533141373155003 * np.std(xs, ddof=1)
                          / math.sqrt(n))
        f_m = _kde_at(xs, est)
        se = (float(1.0 / (2.0 * f_m * math.sqrt(n)))
              if np.isfinite(f_m) and f_m > 0 else float("nan"))
    else:
        se_normal = se = f_m = float("nan")

    zc = _z(1 - alpha / 2)
    lo = est - zc * se if np.isfinite(se) else float("nan")
    hi = est + zc * se if np.isfinite(se) else float("nan")
    elo, ehi, cover = _order_statistic_ci(xs, alpha)

    verified = beaten = None
    if alternatives is not None:
        alts = np.asarray(alternatives, dtype=float).ravel()
        # under single-peaked preferences a voter prefers whichever
        # platform lies nearer their own ideal point
        wins = []
        for a in alts:
            for_med = int(np.sum(np.abs(x - est) < np.abs(x - a)))
            for_alt = int(np.sum(np.abs(x - a) < np.abs(x - est)))
            wins.append(for_med >= for_alt)
        beaten = np.asarray(wins, dtype=bool)
        verified = bool(beaten.all())

    out = RichResult(
        title=_METHOD,
        summary_lines=[
            ("Median (x*)", est),
            ("SE (density)", se),
            ("SE (normal-only)", se_normal),
            ("n", n),
        ],
        interpretation=(
            "With single-peaked preferences in one dimension the Condorcet "
            f"winner is x* = {est:.4f}, the median ideal point, not the mean "
            f"({float(np.mean(x)):.4f})."
        ),
        payload={
            "estimate": est,
            "mean": float(np.mean(x)),
            "se": se,
            "se_normal": se_normal,
            "density_at_median": f_m,
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_exact_lower": elo,
            "ci_exact_upper": ehi,
            "exact_coverage": cover,
            "median_interval": interval,
            "unique_winner": unique,
            "condorcet_verified": verified,
            "alternatives_beaten": beaten,
            "check_is_definitional": alternatives is not None,
            "n": n,
            "method": _METHOD,
        },
    )
    if not unique:
        out.warnings.append(
            f"With an even electorate (n = {n}) every point in "
            f"[{interval[0]:.6g}, {interval[1]:.6g}] is a Condorcet winner. "
            "The reported estimate is the midpoint, which is a convention "
            "rather than a result."
        )
    if np.isfinite(se) and np.isfinite(se_normal) and se > 0:
        ratio = se_normal / se
        if ratio > 1.15 or ratio < 0.87:
            out.warnings.append(
                f"The normality-based standard error is {ratio:.2f} times the "
                "density-based one, so this electorate is far from normal "
                "and the 1.2533 formula should not be quoted for it."
            )
    if verified is False:
        out.warnings.append(
            "The median did not beat every supplied alternative under "
            "distance-based preferences. That should be arithmetically "
            "impossible on one dimension, so this indicates a defect rather "
            "than a finding about the electorate."
        )
    return out


def condorcet_winner(utility, platforms=None):
    """Pairwise-majority winner from a utility matrix, or None on a cycle.

    Unlike the check inside :func:`median_voter`, this makes no
    assumption about where preferences came from. Give it utilities
    that are not single-peaked and it will find no winner, which is the
    Condorcet paradox rather than a bug: with three voters ranking
    three options A > B > C, B > C > A and C > A > B, every option
    loses a pairwise contest to some other and the majority relation
    runs in a circle.

    Parameters
    ----------
    utility : array-like, shape (n_voters, n_options)
        Utility each voter derives from each option. Only the ordering
        within a row matters.
    platforms : array-like, optional
        Labels or positions for the options.

    Returns
    -------
    RichResult with ``winner_index``, ``winner``, ``exists``,
    ``net_wins``, ``beats`` (the pairwise majority matrix), ``cyclic``.
    """
    U = np.atleast_2d(np.asarray(utility, dtype=float))
    if U.ndim != 2:
        raise ValueError(f"utility must be 2-D; got shape {U.shape}.")
    nv, m = U.shape
    if m < 2:
        raise ValueError("need at least two options.")
    beats = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            beats[i, j] = int(np.sum(U[:, i] > U[:, j])) > \
                int(np.sum(U[:, j] > U[:, i]))
    # a Condorcet winner beats every other option
    wins = beats.sum(axis=1)
    idx = [i for i in range(m) if wins[i] == m - 1]
    exists = len(idx) == 1
    w = idx[0] if exists else None
    labels = (np.arange(m) if platforms is None
              else np.asarray(platforms).ravel())
    out = RichResult(
        title="Condorcet winner by pairwise majority",
        summary_lines=[
            ("Winner", labels[w] if exists else "none (cycle)"),
            ("Options", m),
            ("Voters", nv),
        ],
        payload={
            "winner_index": w,
            "winner": (labels[w] if exists else None),
            "estimate": (float(labels[w]) if exists
                         and np.issubdtype(np.asarray(labels).dtype,
                                           np.number) else np.nan),
            "exists": exists,
            "cyclic": not exists,
            "net_wins": wins,
            "beats": beats,
            "n": nv,
            "method": "Condorcet winner by pairwise majority",
        },
    )
    if not exists:
        out.warnings.append(
            "No Condorcet winner: the pairwise-majority relation does not "
            "have a single option beating all others. Preferences are not "
            "single-peaked on any one dimension, so the median voter "
            "theorem does not apply to this profile."
        )
    return out


mdvtr = median_voter


def cheatsheet():
    return (
        "mdvtr: median voter theorem -- x* = median(x_i*) in 1D, with a "
        "density-based standard error rather than the normality-only 1.2533"
    )


# CANONICAL TEST
# >>> r = median_voter([1.0, 2.0, 3.0, 4.0, 100.0])
# >>> assert abs(r["estimate"] - 3.0) < 1e-9
