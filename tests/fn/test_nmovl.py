"""nmovl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmovl import party_overlap


def test_nmovl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        party_overlap(data=None)
