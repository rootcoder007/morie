# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Knowledge distillation: student matches softened teacher outputs."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_knowledge_distillation"]

_METHOD = "Knowledge distillation loss"


def _softmax(z, T=1.0):
    z = np.asarray(z, dtype=float) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def geron_knowledge_distillation(teacher, student, X=None, y=None, T=2.0, alpha=0.5):
    """
    Knowledge distillation: student matches softened teacher outputs.

    Formula: L = alpha*CE(y, student) + (1-alpha)*T^2*KL(softmax(teach/T), softmax(stu/T))

    Two things carry the "dark knowledge": the temperature and the
    ``T^2``.

    Dividing the logits by ``T > 1`` flattens both distributions, which
    exposes the teacher's relative ranking of the *wrong* classes -- a 3
    that looks slightly like an 8 tells the student more than the hard
    label ever does.  At ``T = 1`` nothing is softened and the term is
    ordinary cross-entropy against the teacher.

    The ``T^2`` factor is the correction that makes the two terms
    comparable: softening scales the gradient of the KL term by
    ``1/T^2``, so without it a larger temperature would silently shrink
    the distillation gradient to nothing and ``alpha`` would stop
    meaning what it says.

    ``teacher`` and ``student`` are logits, or callables applied to
    ``X``.  ``y`` are the hard labels; without them ``alpha`` must be 0
    (pure distillation).

    Parameters
    ----------
    teacher, student : array-like, shape (m, C), or callable
        Logits, or ``f(X) -> logits``.
    X : array-like, optional
        Passed to the callables.
    y : array-like of int, shape (m,), optional
        Hard labels.
    T : float
        Temperature, ``T > 0``.
    alpha : float
        Weight on the hard-label term, in [0, 1].

    Returns
    -------
    result : RichResult
        Keys: loss, ce_loss, kl_loss, teacher_probs, student_probs,
        agreement, estimate, n, method.

    Examples
    --------
    A student that matches the teacher exactly has zero KL, so the loss
    is the cross-entropy alone.  With identical logits ``[2, 0, 0]`` and
    label 0, ``CE = -log(softmax([2,0,0])[0]) = 0.239544766``:

    >>> logits = [[2.0, 0.0, 0.0]]
    >>> r = geron_knowledge_distillation(logits, logits, y=[0], T=2.0, alpha=0.5)
    >>> round(r["kl_loss"], 12)
    0.0
    >>> round(r["ce_loss"], 9)
    0.239544766
    >>> round(r["loss"], 9)
    0.119772383

    Pure distillation ignores the labels entirely:

    >>> p = geron_knowledge_distillation([[2.0, 0.0]], [[0.0, 0.0]], T=1.0, alpha=0.0)
    >>> round(p["loss"], 9) == round(p["kl_loss"], 9)
    True

    The KL at T=1 from softmax([2,0]) to the uniform distribution is
    ``0.880797*log(0.880797/0.5) + 0.119203*log(0.119203/0.5)``:

    >>> round(p["kl_loss"], 9)
    0.327813325

    Raising the temperature softens the teacher toward uniform:

    >>> hot = geron_knowledge_distillation([[4.0, 0.0]], [[0.0, 0.0]], T=10.0, alpha=0.0)
    >>> [round(float(v), 6) for v in hot["teacher_probs"][0]]
    [0.598688, 0.401312]

    Callables are accepted, with X supplied:

    >>> c = geron_knowledge_distillation(lambda Z: [[2.0, 0.0]], lambda Z: [[2.0, 0.0]],
    ...                                  X=[[1.0]], T=1.0, alpha=0.0)
    >>> round(c["loss"], 12)
    0.0

    References
    ----------
    Géron Ch 17
    """
    temp = float(T)
    if not np.isfinite(temp) or temp <= 0:
        raise ValueError(f"geron_knowledge_distillation: T must be positive and finite, got {T!r}")
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError(f"geron_knowledge_distillation: alpha must lie in [0, 1], got {alpha!r}")

    def _logits(obj, name):
        if callable(obj):
            if X is None:
                raise ValueError(f"geron_knowledge_distillation: {name} is a callable but X was not supplied")
            obj = obj(X)
        arr = np.atleast_2d(np.asarray(obj, dtype=float))
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"geron_knowledge_distillation: {name} logits must be a non-empty (m, C) array")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_knowledge_distillation: {name} logits contain non-finite values")
        return arr

    tl = _logits(teacher, "teacher")
    sl = _logits(student, "student")
    if tl.shape != sl.shape:
        raise ValueError(
            f"geron_knowledge_distillation: teacher logits have shape {tl.shape} but student's are {sl.shape}"
        )
    m, C = tl.shape

    pt = _softmax(tl, temp)
    ps = _softmax(sl, temp)
    # KL(p_teacher || p_student), summed over classes, averaged over rows.
    log_ratio = np.log(np.clip(pt, 1e-300, None)) - np.log(np.clip(ps, 1e-300, None))
    kl = float(np.mean(np.sum(pt * log_ratio, axis=1)))
    kl_term = temp * temp * kl

    if y is None:
        if a != 0.0:
            raise ValueError(
                f"geron_knowledge_distillation: alpha={a} weights the hard-label term but y was not supplied"
            )
        ce = 0.0
    else:
        yy = np.asarray(y).ravel()
        if yy.size != m:
            raise ValueError(f"geron_knowledge_distillation: y has {yy.size} entries but there are {m} rows")
        if np.any(yy < 0) or np.any(yy >= C):
            raise ValueError(f"geron_knowledge_distillation: labels must lie in 0..{C - 1}, got {np.unique(yy).tolist()}")
        p_hard = _softmax(sl, 1.0)
        ce = float(np.mean(-np.log(np.clip(p_hard[np.arange(m), yy.astype(int)], 1e-300, None))))

    loss = a * ce + (1.0 - a) * kl_term
    agreement = float(np.mean(np.argmax(tl, axis=1) == np.argmax(sl, axis=1)))

    return RichResult(
        title="Knowledge distillation",
        summary_lines=[
            ("Temperature", temp),
            ("alpha", a),
            ("Cross-entropy term", ce),
            ("Distillation term", kl_term),
            ("Total loss", loss),
        ],
        interpretation=(
            "The T^2 factor keeps the two terms comparable: softening scales the KL gradient by "
            "1/T^2, so without it a hotter teacher would quietly stop contributing."
        ),
        payload={
            "loss": loss,
            "ce_loss": ce,
            "kl_loss": kl_term,
            "kl_raw": kl,
            "teacher_probs": pt,
            "student_probs": ps,
            "agreement": agreement,
            "T": temp,
            "alpha": a,
            "estimate": loss,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkd: distillation loss alpha*CE + (1-alpha)*T^2*KL(teacher_T || student_T)"
