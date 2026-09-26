"""zxwmr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxwmr import wavelet_mra_sp


def test_zxwmr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wavelet_mra_sp(data=None)
