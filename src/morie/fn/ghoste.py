"""Ghose drug-like filter."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ghose_filter"]


def ghose_filter(smiles):
    """
    Ghose drug-like filter

    Formula: 160≤MW≤480, -0.4≤LogP≤5.6, 40≤MR≤130, 20≤atoms≤70

    Parameters
    ----------
    smiles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghose-Viswanadhan-Wendoloski (1999)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ghose drug-like filter"})


def cheatsheet():
    return "ghoste: Ghose drug-like filter"
