"""trmtd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trmtd import trmtd


def test_trmtd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trmtd()
