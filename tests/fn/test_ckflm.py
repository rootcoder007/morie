"""ckflm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ckflm import ckflm


def test_ckflm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ckflm()
