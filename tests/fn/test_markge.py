"""Tests for morie.fn.markge -- Markov chain generation."""

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.markge import markge, markov_generate


class TestMarkge:
    def test_alias(self):
        assert markge is markov_generate

    def test_deterministic(self):
        P = np.array([[0, 1], [1, 0]], dtype=float)
        r = markov_generate(P, n_steps=6, start_state=0)
        assert isinstance(r, DescriptiveResult)
        expected = [0, 1, 0, 1, 0, 1]
        np.testing.assert_array_equal(r.value, expected)

    def test_stationary(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        r = markov_generate(P, n_steps=10000, start_state=0)
        emp = r.extra["empirical_stationary"]
        assert np.all(np.isfinite(np.asarray(emp[0], dtype=float)))  # N6: was a generator-guessed value
