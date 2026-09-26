"""ptcsr re-exports the real csr_test from mrkcsr."""

from morie.fn.mrkcsr import csr_test as canonical
from morie.fn.ptcsr import csr_test


def test_ptcsr_is_the_canonical_implementation():
    assert csr_test is canonical
