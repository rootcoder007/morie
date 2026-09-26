"""tmcsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmcsp import tmcsp


def test_tmcsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmcsp()
