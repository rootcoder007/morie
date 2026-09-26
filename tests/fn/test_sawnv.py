"""sawnv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawnv import sawnv


def test_sawnv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawnv()
