# morie.fn -- function file (rootcoder007/morie)
"""Le Cam's posterior inequality."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lecam", "ghosal_lecam_lemma"]


def lecam(dtv, p0_phi, prior_mass, integral):
    """Le Cam's bound on the posterior mass of an alternative set.

    The bound decomposes the posterior mass of V into three separate
    things that can each be controlled on their own: how far the truth
    is from the set U in total variation, how often the test errs under
    the truth, and how much prior mass sits on U.  The last term
    divides by Pi(U), which is why a prior that starves the
    neighbourhood of the truth destroys the bound no matter how good
    the test is.

    Formula: P_0 Pi(V | X) <= d_TV(P_0, P_U) + P_0 phi
                              + (1/Pi(U)) int_V P(1 - phi) dPi(P)

    Parameters
    ----------
    dtv : float
        d_TV(P_0, P_U), in [0, 1].
    p0_phi : float
        P_0 phi, the type I error, in [0, 1].
    prior_mass : float
        Pi(U), strictly positive.
    integral : float
        int_V P(1 - phi) dPi(P), non-negative.

    Returns
    -------
    RichResult
        ``bound``, ``term_tv``, ``term_test``, ``term_prior``,
        ``informative`` (1 when the bound is below 1).

    References
    ----------
    Ghosal & van der Vaart (2017), Fundamentals of Nonparametric
    Bayesian Inference, Lemma 6.46 (Le Cam), stated there as
    P_0 Pi(V | X) <= d_TV(P_0, P_U) + P_0 phi + (1/Pi(U)) int_V
    P(1 - phi) dPi(P), for any measurable sets U, V, test phi and
    measure P_0.  Read from the copy of the book held in the corpus.
    NOTE: the worklist filed this under "Appendix D"; in the book it is
    Lemma 6.46 in Section 6.8.2, not an appendix result.
    """
    dtv = float(dtv)
    p0 = float(p0_phi)
    pm = float(prior_mass)
    it = float(integral)
    if not 0.0 <= dtv <= 1.0:
        raise ValueError("dtv must lie in [0, 1]")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("P_0 phi must lie in [0, 1]")
    if pm <= 0.0:
        raise ValueError("the prior mass Pi(U) must be positive")
    if it < 0.0:
        raise ValueError("the integral must be non-negative")
    t3 = it / pm
    b = dtv + p0 + t3
    return RichResult(payload={
        "bound": b, "term_tv": dtv, "term_test": p0, "term_prior": t3,
        "informative": 1.0 if b < 1.0 else 0.0,
        "method": "Le Cam posterior inequality, Ghosal Lemma 6.46"})


ghosal_lecam_lemma = lecam


def cheatsheet():
    return "gh_ap_d2: Pi(V|X) <= d_TV + P0 phi + (1/Pi(U)) int_V P(1-phi) dPi"
