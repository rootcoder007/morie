# morie.fn -- function file (rootcoder007/morie)
"""Area under ROC curve with R-style verbose result."""

from ._richresult import RichResult


def _bench(auc):
    if auc >= 0.9:
        return "excellent"
    if auc >= 0.8:
        return "good"
    if auc >= 0.7:
        return "fair"
    if auc > 0.5:
        return "poor"
    return "no better than chance"


def aurroc(y_true, score):
    r"""Area under the ROC curve, ``AUC = P(S+ > S-) + P(S+ = S-)/2`` (Mann-Whitney form).

    For binary ``y_true`` (1 positive, 0 negative) the empirical AUC equals
    the Mann-Whitney statistic ``U / (n1 n0)`` with ties counted one half
    (Bamber 1975; Hanley and McNeil 1982). Range [0, 1]: 0.5 is chance, 1
    perfect, 0 perfectly inverted. ``float(result)`` gives the AUC.

    References
    ----------
    Bamber, D. (1975). The area above the ordinal dominance graph and the
    area below the receiver operating characteristic graph. *Journal of
    Mathematical Psychology* 12, 387-415.
    Hanley, J. A. and McNeil, B. J. (1982). The meaning and use of the area
    under a receiver operating characteristic (ROC) curve. *Radiology* 143,
    29-36.

    Examples
    --------
    >>> round(float(aurroc([0, 0, 1, 1, 0, 1], [0.1, 0.4, 0.35, 0.8, 0.4, 0.9])), 12)
    0.777777777778
    """
    y = [int(v) for v in (y_true.tolist() if hasattr(y_true, "tolist") else y_true)]
    s = [float(v) for v in (score.tolist() if hasattr(score, "tolist") else score)]
    pos = [b for a, b in zip(y, s) if a == 1]
    neg = [b for a, b in zip(y, s) if a == 0]
    if not pos or not neg:
        raise ValueError("both classes are needed for the AUC")
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    auc = wins / (len(pos) * len(neg))
    warnings = []
    if min(len(pos), len(neg)) < 10:
        warnings.append(
            f"class imbalance: {len(pos)} positives, {len(neg)} negatives - AUROC unstable; "
            "consider a stratified bootstrap CI."
        )
    bench = _bench(auc)
    return RichResult(
        title="Area under ROC curve",
        summary_lines=[
            ("AUROC", auc),
            ("Benchmark", bench),
            ("n total", len(y)),
            ("n positive (y=1)", len(pos)),
            ("n negative (y=0)", len(neg)),
        ],
        warnings=warnings,
        interpretation=f"AUROC={auc:.3f} -> {bench}.",
        payload={"value": auc, "statistic": auc, "benchmark": bench, "n_positive": len(pos), "n_negative": len(neg)},
    )


def cheatsheet() -> str:
    return "aurroc: aurroc(y_true, score) -> AUC = P(S+ > S-) + P(tie)/2 (Mann-Whitney)."
