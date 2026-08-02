# morie.fn -- function file (rootcoder007/morie)
"""ideal()-style Bayesian roll-call scaling on an encoded rollcall object."""

from . import _array_core as np

from ._richresult import RichResult
from .mcmpp import mcmcpack_irt

__all__ = ["pscl_ideal"]


def pscl_ideal(rollcall_obj, n_iter=2000, burnin=500, seed=0, polarity_idx=None, drop_lopsided=True):
    r"""Run the Albert Gibbs IRT on a :mod:`morie.fn.pscrc` rollcall object.

    Mirrors the pscl ``ideal()`` workflow: take the recoded vote
    matrix, drop the lopsided roll calls the screen flagged (their
    near-unanimity carries no spatial information), and hand the rest
    to the probit-IRT sampler. Accepts either the RichResult from
    ``pscl_rollcall`` or a plain dict with the same keys.

    Parameters
    ----------
    rollcall_obj : mapping
        Output of :func:`morie.fn.pscrc.pscl_rollcall` (needs
        ``votes`` and ``keep``).
    n_iter, burnin, seed, polarity_idx :
        Passed to :func:`morie.fn.mcmpp.mcmcpack_irt`.
    drop_lopsided : bool, default True

    Returns
    -------
    RichResult
        Same keys as ``mcmcpack_irt`` plus ``n_rollcalls_used`` and
        ``n_rollcalls_dropped``.

    References
    ----------
    Jackman, S. (2024). *pscl: Political Science Computational
    Laboratory* (the ideal() workflow this mirrors).

    Clinton, J., Jackman, S. & Rivers, D. (2004). The statistical
    analysis of roll call data. *APSR*, 98(2), 355-370.
    """
    try:
        votes = np.asarray(rollcall_obj["votes"], dtype=float)
        keep = np.asarray(rollcall_obj["keep"], dtype=bool)
    except (KeyError, TypeError) as exc:
        raise ValueError("rollcall_obj must carry 'votes' and 'keep' (see pscrc).") from exc

    used = votes[:, keep] if drop_lopsided else votes
    if used.shape[1] < 2:
        raise ValueError("fewer than 2 roll calls survive the lopsidedness screen.")
    out = mcmcpack_irt(used, n_iter=n_iter, burnin=burnin, seed=seed, polarity_idx=polarity_idx)
    payload = dict(out)
    payload["n_rollcalls_used"] = int(used.shape[1])
    payload["n_rollcalls_dropped"] = int(votes.shape[1] - used.shape[1]) if drop_lopsided else 0
    payload["method"] = "ideal()-style scaling: pscrc screen + Albert Gibbs IRT"
    return RichResult(payload=payload)


def cheatsheet():
    return "pscli: pscrc rollcall -> drop lopsided -> mcmpp Gibbs IRT"
