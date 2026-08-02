# morie.fn -- function file (rootcoder007/morie)
"""MAP theta -- alias entry point."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["theta_map"]


def theta_map(X, items, prior=(0.0, 1.0)):
    """Maximum a posteriori theta for one or many response patterns,
    over a response MATRIX. One implementation: each row goes to
    :func:`morie.fn.mapth.map_theta_estimator`.

    Despite the neighbouring module name, this is the MODE and
    ``morie.fn.theteap.theta_eap`` is the MEAN -- they differ
    whenever the posterior is skewed, which for short tests and
    extreme patterns is most of the time. Both are returned here so
    the difference is visible rather than a matter of which module
    got called.

    References
    ----------
    Mislevy, R. J. (1986), "Bayes modal estimation in item response
    models", *Psychometrika* 51:177-195. Samejima (1969).
    """
    from .mapth import map_theta_estimator

    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    it = np.atleast_2d(np.asarray(items, dtype=float))
    if it.shape[0] != Xm.shape[1]:
        it = it.T
    if it.shape[0] != Xm.shape[1]:
        raise ValueError(
            f"items has {it.shape[0]} rows for {Xm.shape[1]} item columns.")
    a = it[:, 0]
    b = it[:, 1]
    c = it[:, 2] if it.shape[1] > 2 else None
    thetas = np.empty(Xm.shape[0])
    ses = np.empty(Xm.shape[0])
    for i in range(Xm.shape[0]):
        o = map_theta_estimator(Xm[i], a=a, b=b, c=c, prior=prior)
        thetas[i] = o["theta"]
        ses[i] = o["se"]
    return RichResult(payload={
        "theta": thetas if thetas.size > 1 else float(thetas[0]),
        "se": ses if ses.size > 1 else float(ses[0]),
        "n_examinees": int(Xm.shape[0]), "n_items": int(Xm.shape[1]),
        "prior_mean": float(prior[0]), "prior_sd": float(prior[1]),
        "mode_not_mean": "this is the posterior MODE; theta_eap is the "
                         "MEAN, and they differ whenever the posterior is "
                         "skewed -- short tests and extreme patterns",
        "alias_of": "morie.fn.mapth.map_theta_estimator",
        "method": "MAP theta over a response matrix (Mislevy 1986)"})


def cheatsheet():
    return "theteap2: MAP over a matrix -- the MODE, where theteap is the MEAN"
