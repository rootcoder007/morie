"""Tests for morie.fn.inclusion_exclusion_3: values recomputed from first principles."""

from morie.fn.inclusion_exclusion_3 import inclusion_exclusion_3


def test_against_an_explicit_sample_space():
    # 8 atoms of three events with probabilities; P(A or B or C) = 1 - P(none)
    atoms = {
        (a, b, c): w
        for (a, b, c), w in zip(
            [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], [0.1, 0.15, 0.05, 0.2, 0.1, 0.12, 0.08, 0.2]
        )
    }
    P = lambda f: sum(w for k, w in atoms.items() if f(k))  # noqa: E731
    r = inclusion_exclusion_3(
        P(lambda k: k[0]),
        P(lambda k: k[1]),
        P(lambda k: k[2]),
        P(lambda k: k[0] and k[1]),
        P(lambda k: k[0] and k[2]),
        P(lambda k: k[1] and k[2]),
        P(lambda k: k[0] and k[1] and k[2]),
    )
    assert abs(r["p_or"] - (1 - atoms[(0, 0, 0)])) < 1e-15
