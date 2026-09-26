"""wqinf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqinf import wqinf


def test_wqinf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqinf()
