"""zsnni re-exports the real natural_neighbor from sintf."""

from morie.fn.sintf import natural_neighbor as canonical
from morie.fn.zsnni import natural_neighbor


def test_zsnni_is_the_canonical_implementation():
    assert natural_neighbor is canonical
