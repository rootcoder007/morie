"""zxvms is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zxvms import von_mises_sp


def test_zxvms_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        von_mises_sp(data=None)
