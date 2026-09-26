"""finfis re-exports the real fisher_information from fient."""

from morie.fn.fient import fisher_information as canonical
from morie.fn.finfis import fisher_information


def test_finfis_is_the_canonical_implementation():
    assert fisher_information is canonical
