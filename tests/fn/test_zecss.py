"""zecss is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zecss import cusum_spatial


def test_zecss_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cusum_spatial(data=None)
