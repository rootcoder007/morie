"""ubcmm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubcmm import ubcmm


def test_ubcmm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubcmm()
