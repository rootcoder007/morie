"""jrqbq re-exports the real jarque_bera from jarber."""

from morie.fn.jarber import jarque_bera as canonical
from morie.fn.jrqbq import jarque_bera


def test_jrqbq_is_the_canonical_implementation():
    assert jarque_bera is canonical
