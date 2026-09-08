# morie.fn -- function file (rootcoder007/morie)
"""Soft-voting classifier."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["soft_voting_classifier", "geron_voting_soft"]


def soft_voting_classifier(probabilities, y=None, weights=None,
                           classes=None):
    r"""Average predicted class probabilities across classifiers.

    .. math:: \hat p_c = \sum_m w_m p_{mc}, \qquad
              \hat y = \arg\max_c \hat p_c

    Soft voting generally beats hard voting because it weights a
    confident vote more than a marginal one: a classifier saying 0.95
    should count for more than one saying 0.51, and hard voting throws
    that away. The gain is real only when the probabilities are
    CALIBRATED -- an overconfident member dominates the average for the
    same reason a confident one should, whether or not it is right.
    ``mean_confidence`` and ``agreement`` are returned so an
    overconfident or a unanimous panel is visible.

    Both votes are computed, so the comparison is available rather than
    asserted.

    Parameters
    ----------
    probabilities : array-like, shape (M, n, C)
    y : array-like, shape (n,), optional
    weights : array-like, shape (M,), optional
    classes : sequence, optional

    Returns
    -------
    RichResult
        ``prediction``, ``proba``, ``hard_prediction``, ``agreement``,
        ``mean_confidence``, and accuracies when ``y`` is given.

    References
    ----------
    Geron (2022), *Hands-On Machine Learning*, 3rd ed., chapter 7,
    voting classifiers.

    Examples
    --------
    >>> P = [[[0.9, 0.1]], [[0.4, 0.6]]]
    >>> int(soft_voting_classifier(P)["prediction"][0])
    0
    """
    Q = np.asarray(probabilities, dtype=float)
    if Q.ndim != 3:
        raise ValueError(
            "probabilities must be (M, n, C), got %d dimensions." % Q.ndim
        )
    M, n, C = Q.shape
    if M < 2:
        raise ValueError("need at least 2 classifiers, got %d." % M)
    if np.any(Q < -1e-9) or np.any(Q > 1 + 1e-9):
        raise ValueError("probabilities must lie in [0, 1].")
    sums = Q.sum(axis=2)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError(
            "each classifier's probabilities must sum to 1 over classes."
        )
    w = np.ones(M) / M if weights is None else np.asarray(
        weights, dtype=float
    ).ravel()
    if w.size != M:
        raise ValueError("weights has %d entries for %d members." % (w.size, M))
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero.")
    w = w / w.sum()
    labs = np.arange(C) if classes is None else np.asarray(classes)
    if labs.size != C:
        raise ValueError("classes has %d entries for %d columns." % (labs.size, C))

    proba = np.tensordot(w, Q, axes=(0, 0))
    soft = labs[np.argmax(proba, axis=1)]
    votes = np.argmax(Q, axis=2)
    hard_counts = np.stack([(votes == c).T.astype(float) @ w
                            for c in range(C)], axis=1)
    hard = labs[np.argmax(hard_counts, axis=1)]

    payload = {
        "estimate": soft,
        "prediction": soft,
        "proba": proba,
        "hard_prediction": hard,
        "agreement": float(np.mean(hard_counts.max(axis=1))),
        "mean_confidence": float(np.mean(Q.max(axis=2))),
        "soft_hard_disagreement": float(np.mean(soft != hard)),
        "calibration_note": (
            "soft voting only beats hard voting when the probabilities are "
            "calibrated; an overconfident member dominates the average "
            "regardless of whether it is right"
        ),
        "classes": labs,
        "n_members": int(M),
        "n": int(n),
        "method": "Soft-voting classifier",
    }
    if y is not None:
        yv = np.asarray(y).ravel()
        if yv.size != n:
            raise ValueError("y has %d entries for %d rows." % (yv.size, n))
        memb = np.array([float(np.mean(labs[votes[m]] != yv))
                         for m in range(M)])
        payload.update({
            "soft_error": float(np.mean(soft != yv)),
            "hard_error": float(np.mean(hard != yv)),
            "member_errors": memb,
            "best_member_error": float(memb.min()),
            "mean_member_error": float(w @ memb),
        })
    return RichResult(payload=payload)


def cheatsheet():
    return (
        "hmvts: soft voting with the hard vote alongside, and the "
        "calibration caveat that decides which is better"
    )


#: Catalogue alias for :func:`soft_voting_classifier`.
geron_voting_soft = soft_voting_classifier
