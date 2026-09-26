"""dssprt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dssprt import dssp_secondary


def test_dssprt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dssp_secondary(coords=None)
