"""trnsmh is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trnsmh import transmembrane_topology


def test_trnsmh_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        transmembrane_topology(sequence=None)
