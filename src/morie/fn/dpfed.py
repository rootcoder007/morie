# morie.fn -- function file (rootcoder007/morie)
"""DP federated averaging."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["dp_fedavg"]


def dp_fedavg(client_updates, C=1.0, sigma=1.0, seed=None):
    r"""Aggregate client model updates with user-level differential privacy.

    Each client's update is clipped to L2 norm ``C``, the clipped updates are
    summed, Gaussian noise is added once to the sum, and the result is
    averaged.

    The unit here is the **client**, not the record, and that is the whole
    point of doing it this way. Clipping per client bounds what one
    participant -- with all their data -- can contribute, so the guarantee is
    user-level. Clipping per example inside each client would protect
    individual examples while leaving a client who contributed thousands of
    them fully exposed, which is the distinction
    :func:`~morie.fn.dpunit.dp_unit_definition` exists to make explicit.

    Noise is added **once to the aggregate**, not per client, so the noise
    per client falls as :math:`1/m`. Federated DP therefore gets cheaper with
    more participants, which is the opposite of how per-record DP behaves and
    the reason cross-device federation is viable at all.

    Parameters
    ----------
    client_updates : array-like
        One update vector per client, ``(m, p)``.
    C : float
        Per-client L2 clipping bound.
    sigma : float
        Noise multiplier; the aggregate noise sd is ``sigma * C``.
    seed : int, optional
        Seed.

    Returns
    -------
    RichResult
        ``aggregate``, ``clipped_fraction``, ``noise_sd_per_client``,
        ``n_clients``.

    References
    ----------
    McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning
        differentially private recurrent language models. *ICLR 2018*.

    Examples
    --------
    Per-client noise falls as the number of clients grows -- more
    participants make federated DP cheaper, not costlier.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> small = dp_fedavg(rng.normal(size=(10, 5)) * 0.1, C=1.0, sigma=1.0, seed=0)
    >>> big = dp_fedavg(rng.normal(size=(1000, 5)) * 0.1, C=1.0, sigma=1.0, seed=0)
    >>> bool(big["noise_sd_per_client"] < small["noise_sd_per_client"] / 50)
    True

    One client cannot dominate the aggregate, whatever they send.

    >>> U = rng.normal(size=(50, 4)) * 0.1
    >>> U_bad = U.copy(); U_bad[0] = 1e6
    >>> a = dp_fedavg(U, C=1.0, sigma=0.0, seed=0)["aggregate"]
    >>> b = dp_fedavg(U_bad, C=1.0, sigma=0.0, seed=0)["aggregate"]
    >>> bool(np.linalg.norm(b - a) < 2.0 / 50 * 2)
    True

    >>> dp_fedavg([0.1, 0.2], C=1.0)
    Traceback (most recent call last):
        ...
    ValueError: client_updates must be (m, p): one row per client
    """
    U = np.asarray(client_updates, dtype=float)
    if U.ndim != 2:
        raise ValueError("client_updates must be (m, p): one row per client")
    C = float(C)
    if C <= 0:
        raise ValueError("C must be positive")
    m, p = U.shape
    norms = np.linalg.norm(U, axis=1)
    Uc = U * np.minimum(1.0, C / np.maximum(norms, 1e-12))[:, None]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma * C, p) if sigma > 0 else np.zeros(p)
    agg = (Uc.sum(axis=0) + noise) / m
    return RichResult(
        title="DP federated averaging",
        summary_lines=[("clients", int(m)), ("C", C),
                       ("noise sd / client", float(sigma * C / m))],
        warnings=["the privacy unit here is the CLIENT; clipping per example "
                  "inside a client would leave a heavy contributor exposed"],
        payload={
            "aggregate": agg, "clipped_fraction": float(np.mean(norms > C)),
            "noise_sd_per_client": float(sigma * C / m),
            "noise_sd_aggregate": float(sigma * C),
            "n_clients": int(m), "C": C, "sigma": float(sigma),
            "method": "dp_fedavg",
        },
    )


def cheatsheet():
    return "dpfed: unit is the CLIENT; noise added once to the sum, so per-client noise falls as 1/m"


# compact alias per ledger/NAMING.md
dpfedavg = dp_fedavg
