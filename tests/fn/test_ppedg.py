"""ppedg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppedg import ppedg


def test_ppedg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppedg()
