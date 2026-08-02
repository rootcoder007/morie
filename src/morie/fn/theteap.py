# morie.fn -- function file (rootcoder007/morie)
"""EAP theta -- alias entry point."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["theta_eap"]


def theta_eap(X, items, prior=(0.0, 1.0), n_nodes=61):
    """Expected a posteriori theta for one or many response
    patterns. One implementation: each row is passed to
    :func:`morie.fn.eapth.eap_theta_estimator`, so the two catalogue
    entries cannot drift apart. This entry point adds the
    convenience of a response MATRIX (examinees by items) and an
    ``items`` table, which is the shape test-scoring code actually
    holds.

    ``items`` is an array with columns ``a``, ``b`` and optionally
    ``c`` -- one row per item, matching the columns of ``X``.

    References
    ----------
    Bock, R. D. and Mislevy, R. J. (1982), *Applied Psychological
    Measurement* 6:431-444.
    """
    from .eapth import eap_theta_estimator

    Xm = np.atleast_2d(np.asarray(X, dtype=float))
    it = np.atleast_2d(np.asarray(items, dtype=float))
    if it.shape[0] != Xm.shape[1]:
        it = it.T
    if it.shape[0] != Xm.shape[1]:
        raise ValueError(
            f"items has {it.shape[0]} rows for {Xm.shape[1]} item columns.")
    if it.shape[1] < 2:
        raise ValueError("items needs at least an a and a b column.")
    a = it[:, 0]
    b = it[:, 1]
    c = it[:, 2] if it.shape[1] > 2 else None
    thetas = np.empty(Xm.shape[0])
    ses = np.empty(Xm.shape[0])
    for i in range(Xm.shape[0]):
        o = eap_theta_estimator(Xm[i], a=a, b=b, c=c, prior=prior,
                                n_nodes=n_nodes)
        thetas[i] = o["theta"]
        ses[i] = o["se"]
    return RichResult(payload={
        "theta": thetas if thetas.size > 1 else float(thetas[0]),
        "se": ses if ses.size > 1 else float(ses[0]),
        "n_examinees": int(Xm.shape[0]), "n_items": int(Xm.shape[1]),
        "prior_mean": float(prior[0]), "prior_sd": float(prior[1]),
        "n_nodes": int(n_nodes),
        "alias_of": "morie.fn.eapth.eap_theta_estimator",
        "method": "EAP theta over a response matrix (Bock-Mislevy 1982)"})


def cheatsheet():
    return "theteap: EAP over a response MATRIX -- same implementation as eapth"
