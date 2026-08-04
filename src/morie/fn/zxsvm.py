"""Spatial SVM"""

from . import _array_core as np

from ._containers import DescriptiveResult


def spatial_svm(data, *, method="default"):
    """Spatial SVM

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    val = float(np.var(x)) if n > 1 else 0.0
    return DescriptiveResult(
        name="zxsvm",
        value=val,
        extra={"n": n},
    )


spat = spatial_svm


def cheatsheet() -> str:
    return "spatial_svm({}) -> Spatial SVM"


# compact alias per ledger/NAMING.md
spatialsvm = spatial_svm
