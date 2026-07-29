# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.22: Flamingo's weighted multi-dataset training loss."""

import numpy as np

from ._richresult import RichResult
from .km149 import kamath_ch9_flamingo_factorized

__all__ = ["kamath_ch9_flamingo_dataset_mix"]


def kamath_ch9_flamingo_dataset_mix(D_m, lambda_m, x=None, y=None):
    r"""sum_m lambda_m E_{(x,y)~D_m} [ -sum_l log p(y_l|y_<l,x_<=l) ].

    ``D_m`` is one dataset per entry, each a list of sequences of
    per-token conditional probabilities; the inner negative
    log-likelihood is Eq 9.21's, so it is taken from
    ``morie.fn.km149``. The expectation over each dataset is the mean
    across its sequences, and the datasets are then combined with the
    weights ``lambda_m``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.22, printed
    p. 404.

    Examples
    --------
    >>> out = kamath_ch9_flamingo_dataset_mix([[[0.5]], [[0.25]]],
    ...                                       [0.25, 0.75])
    >>> round(out["estimate"], 6)   # 0.25*log2 + 0.75*log4
    1.213008
    """
    datasets = list(D_m)
    lam = np.atleast_1d(np.asarray(lambda_m, dtype=float))
    if len(datasets) == 0:
        raise ValueError("no datasets were given.")
    if lam.size != len(datasets):
        raise ValueError(
            f"{lam.size} weights for {len(datasets)} datasets.")
    if np.any(lam < 0):
        raise ValueError("dataset weights cannot be negative.")
    per_dataset = []
    for m, D in enumerate(datasets):
        seqs = list(D)
        if len(seqs) == 0:
            raise ValueError(f"dataset {m} is empty; its expectation "
                             "is undefined.")
        nlls = [float(kamath_ch9_flamingo_factorized(s)["nll"])
                for s in seqs]
        per_dataset.append(float(np.mean(nlls)))
    total = float(np.dot(lam, per_dataset))
    return RichResult(payload={
        "estimate": total, "per_dataset_nll": per_dataset,
        "weights": [float(v) for v in lam], "n": len(datasets),
        "method": "Flamingo weighted multi-dataset loss "
                  "(Kamath Eq 9.22; per-sequence NLL from km149)"})


def cheatsheet():
    return "km150: sum_m lambda_m * mean per-sequence NLL of dataset m"
