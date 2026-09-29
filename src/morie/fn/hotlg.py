# morie.fn -- function file (rootcoder007/morie)
"""Hotelling model of spatial competition: vote-share equilibria of n candidates on a uniform electorate."""

from __future__ import annotations

from ._richresult import RichResult
from ._rng import random_uniform
from .plucmp import plurality_competition


def _q7(s, p):
    idx = (len(s) - 1) * p
    lo = int(idx)
    h = idx - lo
    return s[lo] if h == 0 else (1 - h) * s[lo] + h * s[lo + 1]


def _eaton_lipsey(n):
    """Equilibrium quantile fractions for a uniform electorate (None when no pure equilibrium exists)."""
    if n <= 2:
        return [0.5] * n
    if n == 3:
        return None
    a = 1.0 / (2 * n - 4)
    return [a, a] + [(2 * j + 3) * a for j in range(n - 4)] + [1.0 - a, 1.0 - a]


def hotelling_model(n_voters: int = 100, n_candidates: int = 2, seed: int = 42) -> RichResult:
    r"""Hotelling-Downs competition of ``n_candidates`` vote-share maximisers over uniform voters.

    Voters are ``n_voters`` Philox uniforms on (0, 1); each votes for the
    nearest candidate (ties split). For a uniform electorate the pure-strategy
    Nash equilibria are (Eaton and Lipsey 1975; Shaked 1975): ``n = 2``, both
    candidates at the median (Hotelling 1929, Downs 1957); ``n = 3``, none;
    ``n >= 4``, configurations with the peripheral candidates paired and no
    candidate's longer half-market exceeding any candidate's whole market. For
    ``n >= 4`` the one returned pairs the outer candidates at ``a`` and ``1 - a``
    with the ``n - 4`` others evenly spaced at ``3a, 5a, ...``, ``a = 1 / (2n -
    4)`` (unique for ``n = 4`` and ``5``). The equilibrium fractions are mapped
    to the sample quantiles of the drawn voters, and the exact finite-electorate
    best responses of :func:`morie.fn.plucmp.plurality_competition` give each
    candidate's largest possible gain from deviating (``max_gain``; 0 means the
    configuration is also a Nash equilibrium of the sampled electorate).

    Parameters
    ----------
    n_voters : number of voters.
    n_candidates : number of candidates.
    seed : Philox seed.

    Returns
    -------
    RichResult
        ``value`` (voter median), ``equilibrium_positions`` (``None`` when
        ``n = 3``), ``fractions`` (population quantiles), ``has_pure_equilibrium``,
        ``shares``, ``max_gain``, ``voter_median``, ``n_voters``, ``n_candidates``.

    References
    ----------
    Hotelling, H. (1929). Stability in competition. *Economic Journal* 39, 41-57.

    Downs, A. (1957). *An Economic Theory of Democracy*. Harper and Row.

    Eaton, B. C. and Lipsey, R. G. (1975). The principle of minimum
    differentiation reconsidered: some new developments in the theory of
    spatial competition. *Review of Economic Studies* 42, 27-49.

    Shaked, A. (1975). Non-existence of equilibrium for the two-dimensional
    three-firms location problem. *Review of Economic Studies* 42, 51-55.

    Examples
    --------
    >>> r = hotelling_model(n_voters=9, n_candidates=2, seed=1)
    >>> r.equilibrium_positions == [r.voter_median] * 2, r.max_gain
    (True, 0.0)
    >>> hotelling_model(n_voters=9, n_candidates=3).has_pure_equilibrium
    False
    >>> hotelling_model(n_candidates=5).fractions
    [0.16666666666666666, 0.16666666666666666, 0.5, 0.8333333333333334, 0.8333333333333334]
    """
    nv, nc = int(n_voters), int(n_candidates)
    if nv < 1 or nc < 1:
        raise ValueError("need at least one voter and one candidate")
    voters = [float(v) for v in random_uniform(nv, seed=seed)]
    s = sorted(voters)
    med = _q7(s, 0.5)
    fr = _eaton_lipsey(nc)
    if fr is None:
        pos, shares, gain = None, None, None
    else:
        pos = [_q7(s, f) for f in fr]
        pc = plurality_competition(voters, pos)
        shares, gain = pc.value, max(pc.extra["gain"])
    return RichResult(
        payload={
            "name": "hotelling_model",
            "value": med,
            "equilibrium_positions": pos,
            "fractions": fr,
            "has_pure_equilibrium": fr is not None,
            "shares": shares,
            "max_gain": gain,
            "voter_median": med,
            "n_voters": nv,
            "n_candidates": nc,
        }
    )


hotlg = hotelling_model


def cheatsheet() -> str:
    return (
        "hotelling_model(n_voters, n_candidates, seed) -> Hotelling-Downs / Eaton-Lipsey equilibria on uniform voters."
    )


# compact alias per ledger/NAMING.md
hotellingmodel = hotelling_model
