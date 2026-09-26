"""enaqi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enaqi import enaqi


def test_enaqi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enaqi()
