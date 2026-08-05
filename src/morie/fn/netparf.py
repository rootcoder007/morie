# morie.fn -- function file (rootcoder007/morie)
"""Network attributable fraction with spillover."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["network_paf"]


def network_paf(y, exposure, network):
    """Attributable fraction split into a direct and a spillover part.

    Under interference an individual's outcome depends on the exposure
    of the people they are connected to, so the ordinary attributable
    fraction -- which assumes each unit's outcome answers only to its
    own exposure -- understates what removing the exposure would do.
    Following the direct/indirect decomposition of Halloran & Hudgens,
    the neighbourhood exposure of unit i is the fraction of its
    neighbours exposed,

        nu_i = sum_j A_ij e_j / sum_j A_ij     (nu_i = 0 if isolated),

    and the outcome is regressed additively on both channels,

        E[Y_i] = b0 + b1 e_i + b2 nu_i.

    The counterfactual mean with nobody exposed and nobody's neighbours
    exposed is ``b0``, so

        PAF = (mean(Y) - b0) / mean(Y)
            = (b1 mean(e) + b2 mean(nu)) / mean(Y),

    which splits exactly into a direct term ``b1 mean(e) / mean(Y)`` and
    a spillover term ``b2 mean(nu) / mean(Y)``.  With ``b2 = 0`` the
    expression collapses to the ordinary attributable fraction, which is
    the identity the decomposition has to satisfy.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome per unit.
    exposure : array-like, shape (n,)
        Own exposure per unit, typically 0/1.
    network : array-like, shape (n, n)
        Non-negative adjacency; row ``i`` gives i's neighbours.  The
        diagonal is ignored, so a unit is not its own neighbour.

    Returns
    -------
    RichResult
        ``estimate`` (total PAF), ``paf``, ``paf_direct``,
        ``paf_spillover``, ``b0``, ``b1``, ``b2``, ``mean_y``,
        ``mean_exposure``, ``mean_nu``, ``n``.

    References
    ----------
    Halloran, M. E. & Hudgens, M. G. (2016).  Dependent happenings: a
    recent methodological review.  Current Epidemiology Reports, 3(4),
    297--305.  doi:10.1007/s40471-016-0086-4
    """
    yv = C.vec(y)
    ev = C.vec(exposure)
    n = len(yv)
    if n == 0:
        raise ValueError("network_paf: y is empty")
    if len(ev) != n:
        raise ValueError("network_paf: y and exposure have different lengths")
    A = C.mat(network)
    if len(A) != n or any(len(r) != n for r in A):
        raise ValueError("network_paf: network must be n by n")
    nu = []
    for i in range(n):
        num = 0.0
        den = 0.0
        for j in range(n):
            if j == i:
                continue
            w = A[i][j]
            if w < 0.0:
                raise ValueError("network_paf: network weights must be non-negative")
            den += w
            num += w * ev[j]
        nu.append(num / den if den > 0.0 else 0.0)
    Xd = [[1.0, ev[i], nu[i]] for i in range(n)]
    beta, _, _, _ = C.lstsq(Xd, yv)
    b0, b1, b2 = beta[0], beta[1], beta[2]
    my = C.mean(yv)
    if my == 0.0:
        raise ValueError("network_paf: mean outcome is zero, PAF undefined")
    me, mn = C.mean(ev), C.mean(nu)
    pd_ = b1 * me / my
    ps = b2 * mn / my
    return RichResult(payload={
        "estimate": pd_ + ps, "paf": pd_ + ps,
        "paf_direct": pd_, "paf_spillover": ps,
        "b0": b0, "b1": b1, "b2": b2,
        "mean_y": my, "mean_exposure": me, "mean_nu": mn, "n": n,
        "method": "Network attributable fraction with spillover"})


def cheatsheet():
    return "netparf: Attributable fraction split into direct and spillover"


networkpaf = network_paf
