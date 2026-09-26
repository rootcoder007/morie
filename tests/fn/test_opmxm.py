"""opmxm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opmxm import opmxm


def test_opmxm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opmxm()
