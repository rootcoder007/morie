r"""PATE: private aggregation of teacher ensembles.

Papernot, N., Abadi, M., Erlingsson, U., Goodfellow, I., & Talwar, K. (2017)
"Semi-supervised Knowledge Transfer for Deep Learning from Private Training
Data", *ICLR 2017*.

Partition the sensitive data, train one *teacher* per partition, and never
release a teacher. To label a record for the *student*, the teachers vote and
the vote is reported with Laplacian noise (equation 1):

.. math:: f(\vec{x}) = \arg\max_j \left\{ n_j(\vec{x})
          + \mathrm{Lap}\left(\tfrac{1}{\gamma}\right) \right\},

with :math:`n_j(\vec{x})` the number of teachers voting for class :math:`j`
and :math:`\gamma` the privacy parameter. Changing one training record can
move one teacher, so the counts on adjacent databases "differ by at most 1 in
at most two locations", and each query is :math:`(2\gamma, 0)`-differentially
private (Theorem 2). The student trains only on these noisy labels, so it
inherits the guarantee and can be published.

**Accounting.** Both analyses in the paper are implemented, and the module
reports both because the gap between them is the paper's point.

*Data-independent* (section 3.2): composing :math:`T` queries of a
:math:`(2\gamma, 0)`-DP mechanism gives

.. math:: \varepsilon = 4T\gamma^2 + 2\gamma\sqrt{2T\ln\tfrac{1}{\delta}},

which "can be rather large" -- the paper's own worked values are
:math:`\gamma = 0.05, T = 1000, \delta = 10^{-6} \Rightarrow \varepsilon
\approx 26` and :math:`\gamma = 0.05, T = 100, \delta = 10^{-5} \Rightarrow
\varepsilon \approx 5.80`, both of which :func:`epsilon_data_independent`
reproduces.

*Data-dependent* (section 3.3): "when the quorum among the teachers is very
strong, the majority outcome has overwhelming likelihood, in which case the
privacy cost is small whenever this outcome occurs". Lemma 4 bounds the
probability that the noisy argmax misses the true winner,

.. math:: \Pr[M(d) \ne j^*] \le \sum_{j \ne j^*}
          \frac{2 + \gamma(n_{j^*} - n_j)}{4\exp(\gamma(n_{j^*} - n_j))},

and Theorem 3 turns that :math:`q` into a moment bound

.. math:: \alpha(l) \le \log\Big((1-q)\big(\tfrac{1-q}{1 - e^{2\gamma}q}
          \big)^{l} + q e^{2\gamma l}\Big),
          \qquad q < \frac{e^{2\gamma} - 1}{e^{4\gamma} - 1},

valid only under that condition on :math:`q`; outside it the mechanism falls
back on Theorem 2's :math:`\alpha(l) \le 2\gamma^2 l(l+1)`. The accountant
takes the smaller of the two at each step, adds them across steps
(Theorem 1, composability), and converts to :math:`(\varepsilon, \delta)` by
the tail bound :math:`\delta = \min_\lambda \exp(\alpha(\lambda) -
\lambda\varepsilon)`, i.e.
:math:`\varepsilon = \min_\lambda (\alpha(\lambda) + \ln(1/\delta))/\lambda`,
over integer :math:`\lambda` as the paper does.

The two accountings are not ordered. Once the quorum is decisive the
data-dependent bound *falls* as :math:`\gamma` grows -- more signal, a
near-certain outcome, little to leak -- while the closed form of section 3.2
grows without limit; on 40 queries with a 24-to-1 vote they run 3.24 against
3.44 at :math:`\gamma = 0.05` and 1.47 against 220.7 at
:math:`\gamma = 1`. Where the vote is split, Theorem 3's condition fails,
the accountant falls back on Theorem 2 for every query, and the closed form
can be the tighter of the two. The reported ``epsilon`` is therefore the
minimum of both.

**Student.** :func:`pate` labels the student's unlabelled data with the noisy
aggregate and, given a learner, trains the student on those labels. The
semi-supervised GAN variant of section 4 is **not** implemented; the labels
and the accounting are, and the student here is trained on the labelled
queries alone.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["pate", "private_aggregation", "pate_aggregate",
           "noisy_argmax", "teacher_votes", "epsilon_data_independent",
           "moments_accountant", "lemma4_bound", "theorem3_moment"]


def teacher_votes(teacher_predicts, rows, n_classes=None):
    r"""The vote histogram :math:`n_j(\vec{x})` for each record.

    ``teacher_predicts`` is a sequence of callables mapping rows to labels
    (or to probability vectors, in which case the argmax is the vote).
    """
    teachers = list(teacher_predicts)
    if not teachers:
        raise ValueError("pate: at least one teacher is needed")
    votes = None
    for predict in teachers:
        out = predict(rows)
        labels = []
        for v in out:
            if isinstance(v, (list, tuple)):
                labels.append(max(range(len(v)), key=lambda t: v[t]))
            else:
                labels.append(int(v))
        if votes is None:
            k = n_classes or (max(labels) + 1)
            votes = [[0] * k for _ in rows]
        for i, lab in enumerate(labels):
            if lab >= len(votes[i]):
                for row in votes:
                    row.extend([0] * (lab + 1 - len(row)))
            votes[i][lab] += 1
    return votes


def noisy_argmax(counts, gamma, rng=None, seed=0):
    r"""Equation 1: :math:`\arg\max_j \{n_j + \mathrm{Lap}(1/\gamma)\}`.

    One independent Laplace draw per class, scale :math:`1/\gamma`. Larger
    :math:`\gamma` means less noise and less privacy.
    """
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("pate: gamma must be positive")
    rng = np.random.default_rng(seed) if rng is None else rng
    best, arg = None, 0
    for j, n in enumerate(counts):
        u = rng.random() - 0.5
        lap = -math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u)) / gamma
        v = n + lap
        if best is None or v > best:
            best, arg = v, j
    return arg


def epsilon_data_independent(T, gamma, delta):
    r"""Section 3.2: :math:`4T\gamma^2 + 2\gamma\sqrt{2T\ln(1/\delta)}`.

    The composition of :math:`T` queries, each :math:`(2\gamma, 0)`-DP,
    without looking at the votes.
    """
    T = float(T)
    gamma = float(gamma)
    delta = float(delta)
    if T < 0 or gamma <= 0:
        raise ValueError("pate: need T >= 0 and gamma > 0")
    if not 0.0 < delta < 1.0:
        raise ValueError("pate: delta must lie in (0, 1)")
    return 4.0 * T * gamma ** 2 + 2.0 * gamma * math.sqrt(
        2.0 * T * math.log(1.0 / delta))


def lemma4_bound(counts, gamma):
    r"""Lemma 4: an upper bound on :math:`\Pr[M(d) \ne j^*]`.

    :math:`\sum_{j \ne j^*} \frac{2 + \gamma(n_{j^*} - n_j)}
    {4\exp(\gamma(n_{j^*} - n_j))}`, with :math:`j^*` the plurality winner.
    The bound can exceed 1 when the quorum is weak; it is a bound, so it is
    returned clipped at 1 and the raw value is available as the second
    element.
    """
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("pate: gamma must be positive")
    n = [float(v) for v in counts]
    if not n:
        raise ValueError("pate: empty vote vector")
    js = max(range(len(n)), key=lambda t: n[t])
    tot = 0.0
    for j in range(len(n)):
        if j == js:
            continue
        gap = gamma * (n[js] - n[j])
        tot += (2.0 + gap) / (4.0 * math.exp(gap))
    return min(tot, 1.0), tot


def theorem3_moment(q, gamma, l):
    r"""Theorem 3's data-dependent moment bound, or ``None`` when its
    condition fails.

    :math:`\alpha(l) \le \log((1-q)(\frac{1-q}{1-e^{2\gamma}q})^{l}
    + q e^{2\gamma l})`, valid for
    :math:`q < \frac{e^{2\gamma}-1}{e^{4\gamma}-1}`. Returning ``None``
    rather than a number outside that range is deliberate: the bound is not
    proved there, and the accountant must fall back on Theorem 2.
    """
    q = float(q)
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("pate: gamma must be positive")
    if q < 0.0:
        raise ValueError("pate: q must be non-negative")
    limit = (math.exp(2.0 * gamma) - 1.0) / (math.exp(4.0 * gamma) - 1.0)
    if q >= limit:
        return None
    if q == 0.0:
        return 0.0
    ratio = (1.0 - q) / (1.0 - math.exp(2.0 * gamma) * q)
    if ratio <= 0.0:
        return None
    return math.log((1.0 - q) * ratio ** l + q * math.exp(2.0 * gamma * l))


def moments_accountant(vote_counts, gamma, delta, lambdas=None,
                       data_dependent=True):
    r"""Compose the per-query moment bounds and convert to
    :math:`(\varepsilon, \delta)`.

    For each query the accountant takes the smaller of Theorem 2's
    :math:`2\gamma^2 l(l+1)` and, when its condition holds, Theorem 3's
    data-dependent bound with :math:`q` from Lemma 4. Theorem 1 adds them
    over queries; the tail bound gives
    :math:`\varepsilon = \min_\lambda (\alpha(\lambda) + \ln(1/\delta))/
    \lambda`.

    ``lambdas`` defaults to the integers 1..8, "a few values of lambda
    (integers up to 8)" as in the paper.
    """
    gamma = float(gamma)
    delta = float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("pate: delta must lie in (0, 1)")
    lams = list(lambdas) if lambdas else list(range(1, 9))
    if not lams or any(l <= 0 for l in lams):
        raise ValueError("pate: lambdas must be positive")
    alpha = dict((l, 0.0) for l in lams)
    used = {"data_dependent": 0, "data_independent": 0}
    for counts in vote_counts:
        q = lemma4_bound(counts, gamma)[0] if data_dependent else 1.0
        for l in lams:
            indep = 2.0 * gamma ** 2 * l * (l + 1)
            dep = theorem3_moment(q, gamma, l) if data_dependent else None
            if dep is not None and dep < indep:
                alpha[l] += dep
                if l == lams[0]:
                    used["data_dependent"] += 1
            else:
                alpha[l] += indep
                if l == lams[0]:
                    used["data_independent"] += 1
    log_inv_delta = math.log(1.0 / delta)
    best = None
    best_l = lams[0]
    for l in lams:
        eps = (alpha[l] + log_inv_delta) / l
        if best is None or eps < best:
            best, best_l = eps, l
    return {"epsilon": best, "lambda": best_l, "alpha": alpha,
            "delta": delta, "queries": len(vote_counts), "used": used}


def pate(teacher_predicts, queries, gamma=0.05, delta=1e-5, n_classes=None,
         student_train_fn=None, student_features=None, seed=0,
         lambdas=None):
    r"""Label ``queries`` by noisy teacher aggregation and account for the
    privacy cost.

    Parameters
    ----------
    teacher_predicts : sequence of callables
        The trained teachers. Each maps a list of records to labels or to
        probability vectors. They are never released.
    queries : sequence
        The student's unlabelled records.
    gamma : float
        The privacy parameter of equation 1; the Laplace scale is
        :math:`1/\gamma`. Smaller means noisier and more private.
    delta : float
        The :math:`\delta` of the :math:`(\varepsilon, \delta)` guarantee.
    n_classes : int, optional
        Number of classes, inferred from the votes when omitted.
    student_train_fn : callable, optional
        ``train_fn(X, y) -> predict_fn``. Given one, the student is trained
        on the noisy labels and returned.
    student_features : sequence, optional
        Features for the student, if they differ from ``queries``.
    seed : int
        Seed for the Laplace draws.
    lambdas : sequence of int, optional
        Moment orders for the accountant.

    Returns
    -------
    RichResult
        ``estimate`` / ``labels`` are the noisy aggregated labels;
        ``clean_labels`` the noiseless plurality (for comparison only -- it
        is NOT private and must not be released); ``votes`` the histograms;
        ``agreement`` the fraction where noise did not change the answer;
        ``epsilon`` is the guarantee, the smaller of ``epsilon_accountant``
        (the moments accountant, data-dependent where Theorem 3 applies)
        and ``epsilon_data_independent`` (section 3.2); both are reported
        because neither dominates. ``accountant`` is the full breakdown and
        ``student`` the trained student when a learner was given.

    Examples
    --------
    Ten teachers labelling 100 student records::

        res = pate(teachers, unlabelled, gamma=0.05, delta=1e-5)
        res["labels"][0], res["epsilon"]

    References
    ----------
    Papernot, Abadi, Erlingsson, Goodfellow & Talwar (2017) ICLR:
    equation 1, Theorems 1-3 and Lemma 4.
    """
    rows = list(queries)
    if not rows:
        raise ValueError("pate: no queries to label")
    votes = teacher_votes(teacher_predicts, rows, n_classes)
    rng = np.random.default_rng(seed)
    labels = [noisy_argmax(v, gamma, rng) for v in votes]
    clean = [max(range(len(v)), key=lambda t: v[t]) for v in votes]
    acct = moments_accountant(votes, gamma, delta, lambdas)
    indep = epsilon_data_independent(len(rows), gamma, delta)
    # Both bounds are valid, so the guarantee is the smaller. They are not
    # ordered in general: the accountant wins by a wide margin once the
    # quorum is decisive, and the closed-form composition of section 3.2
    # can win when it is not.
    eps = min(acct["epsilon"], indep)
    student = None
    if student_train_fn is not None:
        X = list(student_features) if student_features is not None else rows
        if len(X) != len(labels):
            raise ValueError("pate: student_features must be one per query")
        student = student_train_fn(X, labels)
    return RichResult(payload={
        "estimate": labels,
        "labels": labels,
        "clean_labels": clean,
        "votes": votes,
        "agreement": sum(1 for a, b in zip(labels, clean) if a == b) /
                     float(len(labels)),
        "epsilon": eps,
        "epsilon_accountant": acct["epsilon"],
        "epsilon_data_independent": indep,
        "accountant": acct,
        "delta": float(delta),
        "gamma": float(gamma),
        "n_teachers": len(list(teacher_predicts)),
        "n_queries": len(rows),
        "student": student,
        "note": "clean_labels are the noiseless plurality and carry NO "
                "privacy guarantee; the semi-supervised GAN student of "
                "section 4 is not implemented",
        "method": "PATE noisy teacher aggregation (Papernot et al. 2017)",
    })


def cheatsheet():
    return ("pate: private aggregation of teacher ensembles (Papernot et "
            "al. 2017). Teachers trained on disjoint partitions vote; the "
            "student sees only argmax_j {n_j + Lap(1/gamma)} (eq.1), which "
            "is (2 gamma, 0)-DP per query since one record moves one "
            "teacher. Two accountings: data-independent "
            "4 T gamma^2 + 2 gamma sqrt(2 T ln(1/delta)), which reproduces "
            "the paper's 26 and 5.80; and data-dependent, where a strong "
            "quorum makes the majority near-certain, q from Lemma 4 feeds "
            "Theorem 3, the smaller of that and 2 gamma^2 l(l+1) is taken "
            "per query, summed by Theorem 1, and converted by the tail "
            "bound eps = min_lambda (alpha + ln(1/delta))/lambda. The "
            "noiseless plurality is NOT private.")


# compact alias per ledger/NAMING.md
private_aggregation = pate

# name carried over from the generated stub this replaced
pate_aggregate = pate
