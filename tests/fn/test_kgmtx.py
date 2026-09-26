"""kgmtx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgmtx import kriging_matrix


def test_kgmtx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kriging_matrix(values=None, x=None)
