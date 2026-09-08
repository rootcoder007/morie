"""Zero-concentrated differential privacy accounting."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["zcdp"]


def zcdp(mech, rho, delta=1e-6, sensitivity=1.0):
    """
    Zero-concentrated differential privacy (zCDP) accounting

    Formula: D_alpha(M(D) || M(D')) <= rho * alpha for all alpha in (1, inf)

    ``mech`` gives the per-mechanism zCDP parameters rho_i of the
    mechanisms being composed.  Composition is additive (Lemma 1.7), so
    the composed mechanism is rho_total-zCDP with
    ``rho_total = sum(rho_i)``.  ``rho`` is the privacy budget the
    composition is checked against, and is also the budget used to size
    the Gaussian noise.

    The reported quantities are, in the numbering of Bun & Steinke
    (2016):

    * Definition 1.1 -- M is rho-zCDP when
      ``D_alpha(M(x) || M(x')) <= rho * alpha`` for every
      ``alpha in (1, inf)`` and every pair of adjacent inputs.  The
      general form of Definition 1.1 carries an additive ``xi``; the
      ``xi = 0`` case is the one written above and is what this function
      accounts for.
    * Proposition 1.3 -- rho-zCDP implies
      ``(rho + 2 * sqrt(rho * log(1 / delta)), delta)``-differential
      privacy for every ``delta > 0``.
    * Proposition 1.4 -- eps-differential privacy implies
      ``(eps ** 2 / 2)``-zCDP.  Reported as ``rho_from_eps``.
    * Proposition 1.6 / Lemma 2.4 -- the Gaussian mechanism that answers
      a sensitivity-``Delta`` query with noise ``N(0, sigma ** 2)`` is
      ``Delta ** 2 / (2 * sigma ** 2)``-zCDP.  Inverting at the budget
      ``rho`` gives the noise scale ``sigma = Delta / sqrt(2 * rho)``.

    Parameters
    ----------
    mech : array-like
        Per-mechanism zCDP parameters rho_i of the mechanisms being
        composed.  A scalar is treated as a single mechanism.
    rho : float
        Total zCDP privacy budget, ``rho > 0``.
    delta : float, optional
        The delta at which the (eps, delta)-DP statement of
        Proposition 1.3 is reported.  Must satisfy ``0 < delta < 1``.
    sensitivity : float, optional
        Query sensitivity Delta used to size the Gaussian noise.

    Returns
    -------
    result : RichResult
        Keys: rho_total, rho_budget, within_budget, epsilon, delta,
        sigma, rho_from_eps, n_mechanisms, method.

    References
    ----------
    Bun M & Steinke T (2016). Concentrated differential privacy:
    simplifications, extensions, and lower bounds.  Theory of
    Cryptography (TCC 2016-B), 635-658.  arXiv:1605.02065.
    Definition 1.1, Propositions 1.3, 1.4 and 1.6, Lemmas 1.7 and 2.4.
    """
    parts = np.atleast_1d(np.asarray(mech, dtype=float))
    vals = [float(v) for v in parts.tolist()]
    if any(v < 0.0 for v in vals):
        raise ValueError("zCDP parameters rho_i must be non-negative")
    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    delta = float(delta)
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    sensitivity = float(sensitivity)
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be positive")

    # Lemma 1.7: composition of rho_i-zCDP mechanisms is (sum rho_i)-zCDP.
    rho_total = 0.0
    for v in vals:
        rho_total += v
    # Proposition 1.3.
    epsilon = rho_total + 2.0 * np.sqrt(rho_total * np.log(1.0 / delta))
    # Proposition 1.6 inverted at the budget rho.
    sigma = sensitivity / np.sqrt(2.0 * rho)
    # Proposition 1.4, applied to the epsilon just derived.
    rho_from_eps = 0.5 * epsilon * epsilon
    return RichResult(
        payload={
            "rho_total": float(rho_total),
            "rho_budget": rho,
            "within_budget": bool(rho_total <= rho),
            "epsilon": float(epsilon),
            "delta": delta,
            "sigma": float(sigma),
            "rho_from_eps": float(rho_from_eps),
            "n_mechanisms": len(vals),
            "method": "Zero-concentrated DP (Bun & Steinke 2016)",
        }
    )


def cheatsheet():
    return "zcdp: zero-concentrated DP accounting (Bun-Steinke 2016)"
