# morie.fn -- function file (rootcoder007/morie)
"""Matthews correlation coefficient with R-style verbose result."""


def mcc(tp: int, tn: int, fp: int, fn: int):
    """Matthews correlation coefficient for 2x2 confusion matrix.

    Balanced even with imbalanced classes; range [-1, 1].

    Returns RichResult; float(result) yields the scalar.
    """
    from ._richresult import RichResult

    num = tp * tn - fp * fn
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        v = 0.0
        warnings = ["denominator is zero - MCC defined as 0; check class imbalance."]
    else:
        v = float(num / denom)
        warnings = []
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n > 0 else float("nan")
    return RichResult(
        title="Matthews correlation coefficient",
        summary_lines=[
            ("MCC", v),
            ("Accuracy (for context)", accuracy),
            ("TP", tp),
            ("TN", tn),
            ("FP", fp),
            ("FN", fn),
            ("n total", n),
        ],
        warnings=warnings,
        interpretation=(
            f"MCC={v:+.3f}: {abs(v):.2f} magnitude on [0, 1] scale; "
            "negative = predictions inversely correlated with truth."
        ),
        payload={"value": v, "statistic": v, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


def matthews_corrcoef(y_true, y_pred) -> float:
    """MCC from label vectors, in the scikit-learn calling convention.

    This name carries scikit-learn's contract, not morie's: it takes two
    label vectors rather than four confusion-matrix counts, and it
    returns a bare float rather than a RichResult, so the result compares
    and orders like a number. Use :func:`mcc` for the counts form and the
    verbose result.

    Binary labels only. The counts are derived and handed to :func:`mcc`,
    so both entry points share one implementation of the formula.

    References
    ----------
    Matthews, B. W. (1975). Comparison of the predicted and observed
    secondary structure of T4 phage lysozyme. *Biochimica et Biophysica
    Acta*, 405(2), 442-451.
    """
    import numpy as np

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size != yp.size:
        raise ValueError(f"y_true and y_pred must be the same length; got {yt.size}, {yp.size}.")
    if yt.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    labels = np.unique(np.concatenate([yt, yp]))
    if labels.size > 2:
        raise ValueError(f"matthews_corrcoef handles binary labels only; got {labels.size} distinct values.")

    # Map whatever the two labels are onto 0/1 so the counts below are
    # well defined for {-1, 1}, {"neg", "pos"} and so on. With a single
    # label present, everything is the negative class.
    pos = labels[-1] if labels.size == 2 else None
    t = yt == pos if pos is not None else np.zeros(yt.size, dtype=bool)
    p = yp == pos if pos is not None else np.zeros(yp.size, dtype=bool)

    tp = int(np.sum(t & p))
    tn = int(np.sum(~t & ~p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    return float(mcc(tp, tn, fp, fn)["value"])
