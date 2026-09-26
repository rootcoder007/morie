"""zxwlt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxwlt import wavelet_spatial


def test_zxwlt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wavelet_spatial(data=None)
