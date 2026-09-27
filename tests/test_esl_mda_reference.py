"""Mixture discriminant analysis (ESL 12.59-12.61) against mda::mda started from the same responsibilities."""

import math

from morie.fn.eslmda import esl_mda


def test_mda_equals_mda_package():
    i = list(range(1, 91))
    X = [
        [
            math.sin(t) + ((t % 3) == 0) * 1.5 + ((t % 7) < 3) * 0.8,
            math.cos(2 * t) + ((t % 3) == 1) * 1.2 - ((t % 5) < 2) * 0.9,
        ]
        for t in i
    ]
    g = [t % 3 for t in i]
    r = esl_mda(X, g, 2, query=[[0.2, 0.5], [1.5, -0.8], [-0.5, 1.1]])
    ref = (
        (1.47630920950233e-02, 0.3929668448304358, 0.5922700630745410),
        (2.84808169363645e-01, 0.0629460807514707, 0.6522457498848843),
        (1.45626100979666e-06, 0.9891088453292690, 0.0108896984097212),
    )
    # mda iterates its optimal-scoring EM to ~1e-7; ours converges the likelihood to 1e-12
    assert all(abs(a - b) < 1e-6 for pr, rr in zip(r["posterior"], ref) for a, b in zip(pr, rr))
    assert r["prediction"] == [2, 2, 1] and r["converged"]
    tr = esl_mda(X, g, 2)
    conf = [[sum(1 for p_, t in zip(tr["prediction"], g) if p_ == a and t == b) for b in range(3)] for a in range(3)]
    assert conf == [[27, 8, 4], [0, 19, 3], [3, 3, 23]]
