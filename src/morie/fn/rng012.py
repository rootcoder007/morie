# morie.fn -- function file (rootcoder007/morie)
"""Additive signal-plus-noise model (Rangayyan eqs. 3.12-3.14)."""


from math import fsum

from ._rgcore import aslist
from ._richresult import RichResult

__all__ = ["noisemodel", "rangayyan_ch3_signal_plus_noise_model"]


def noisemodel(x, eta):
    """Form the observed signal y = x + eta and its first two moments.

    Rangayyan (2024) eqs. (3.12)-(3.14):
        y(t)  = x(t) + eta(t)                                   (3.12)
        mu_y  = mu_x + mu_eta                                   (3.13)
        sig_y^2 = sig_x^2 + sig_eta^2, IF x and eta uncorrelated (3.14)

    Eq. (3.14) holds only under uncorrelatedness, so the sample
    correlation is computed and reported rather than assumed: compare
    ``variance_additive`` (the eq. 3.14 prediction) against
    ``variance_observed`` to see how far the assumption is from holding
    on this particular pair.
    """
    xs, es = aslist(x), aslist(eta)
    if len(xs) != len(es):
        raise ValueError("signal and noise must have the same length")
    n = len(xs)
    if n == 0:
        raise ValueError("need at least one sample")
    y = [a + b for a, b in zip(xs, es)]
    mx, me = fsum(xs) / n, fsum(es) / n
    vx = fsum((v - mx) ** 2 for v in xs) / n
    ve = fsum((v - me) ** 2 for v in es) / n
    my = fsum(y) / n
    vy = fsum((v - my) ** 2 for v in y) / n
    cov = fsum((a - mx) * (b - me) for a, b in zip(xs, es)) / n
    rho = cov / ((vx * ve) ** 0.5) if vx > 0 and ve > 0 else 0.0
    return RichResult(payload={
        "y": y, "mean_signal": mx, "mean_noise": me,
        "mean_observed": my, "mean_additive": mx + me,
        "variance_observed": vy, "variance_additive": vx + ve,
        "covariance": cov, "correlation": rho, "n": n,
        "method": "Rangayyan (2024) eqs. (3.12)-(3.14)"})


rangayyan_ch3_signal_plus_noise_model = noisemodel  # pre-policy spelling


def cheatsheet():
    return "rng012: additive noise model, Rangayyan eqs. (3.12)-(3.14)"
