"""dkvck is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkvck import dkvck


def test_dkvck_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkvck()
