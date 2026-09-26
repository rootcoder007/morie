"""enspd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enspd import enspd


def test_enspd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enspd()
