"""Valence advantage model (Groseclose)"""

from . import _array_core as np

from ._containers import DescriptiveResult


def valence_model(data, *, method="default"):
    """Valence advantage model (Groseclose)

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return DescriptiveResult(
        name="svvlm",
        value=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


vale = valence_model


def cheatsheet() -> str:
    return "valence_model({}) -> Valence advantage model (Groseclose)"


# compact alias per ledger/NAMING.md
valencemodel = valence_model
