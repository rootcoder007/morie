"""zxcmn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxcmn import circular_mean_sp


def test_zxcmn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        circular_mean_sp(data=None)
