# morie.fn -- function file (rootcoder007/morie)
"""Random utility model (McFadden) for stochastic choice."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["random_utility_model"]


def random_utility_model(V, eps_dist="gumbel", n_draws=20000, seed=0):
    r"""Choice probabilities from :math:`U_{ij} = V_{ij} + \varepsilon_{ij}`.

    With i.i.d. Gumbel errors the probabilities are the closed-form
    conditional logit,

    .. math:: P_{ij} = \frac{e^{V_{ij}}}{\sum_k e^{V_{ik}}},

    (McFadden). With ``eps_dist="normal"`` the multinomial-probit
    probabilities have no closed form and are simulated by Monte
    Carlo -- the frequency of :math:`\arg\max_j (V_{ij} +
    \varepsilon_{ij})` over draws, the accept-reject frequency
    simulator Train describes before introducing GHK.

    Parameters
    ----------
    V : array-like, shape (n, J) or (J,)
        Systematic utilities.
    eps_dist : {"gumbel", "normal"}, default "gumbel"
    n_draws : int, default 20000
        Simulation draws for the normal case.
    seed : int, default 0

    Returns
    -------
    RichResult
        keys: ``probabilities`` (same leading shape as V), ``chosen``
        (argmax of the probabilities), ``eps_dist``, ``method``.

    References
    ----------
    McFadden, D. (1974). Conditional logit analysis of qualitative
    choice behavior. In P. Zarembka (ed.), *Frontiers in
    Econometrics*, Academic Press, 105-142.

    Train, K. E. (2009). *Discrete Choice Methods with Simulation*
    (2nd ed.). Cambridge University Press. Ch. 3 (logit), Ch. 5
    (probit simulation).
    """
    Va = np.atleast_2d(np.asarray(V, dtype=float))
    n, J = Va.shape
    if J < 2:
        raise ValueError("need at least 2 alternatives.")
    if eps_dist not in ("gumbel", "normal"):
        raise ValueError(f"eps_dist must be 'gumbel' or 'normal', got {eps_dist!r}.")

    if eps_dist == "gumbel":
        z = Va - Va.max(axis=1, keepdims=True)
        ez = np.exp(z)
        P = ez / ez.sum(axis=1, keepdims=True)
    else:
        B = int(n_draws)
        if B < 1000:
            raise ValueError(f"n_draws must be at least 1000, got {B}.")
        rng = np.random.default_rng(seed)
        P = np.zeros((n, J))
        for i in range(n):
            u = Va[i] + rng.standard_normal((B, J))
            winners = np.argmax(u, axis=1)
            P[i] = np.bincount(winners, minlength=J) / B

    scalar = np.ndim(V) == 1
    probs = P[0] if scalar else P
    return RichResult(
        payload={
            "probabilities": probs,
            "chosen": int(np.argmax(P[0])) if scalar else np.argmax(P, axis=1),
            "eps_dist": eps_dist,
            "method": f"Random utility model ({eps_dist} errors)",
        }
    )


def cheatsheet():
    return "rndut: Gumbel -> closed-form logit; normal -> simulated probit frequencies"
