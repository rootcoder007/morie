"""wqlvs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqlvs import wqlvs


def test_wqlvs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqlvs()
