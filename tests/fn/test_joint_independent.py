"""Tests for morie.fn.joint_independent: values recomputed from first principles."""

from morie.fn.joint_independent import joint_independent


def test_factorising_and_not():
    px, py = [0.3, 0.7], [0.4, 0.6]
    J = [[a * b for b in py] for a in px]
    r = joint_independent(J)
    assert r["independent"] is True
    assert max(abs(a - b) for a, b in zip(r["marginal_x"], px)) < 1e-15
    assert joint_independent([[0.2, 0.1], [0.1, 0.6]])["independent"] is False
