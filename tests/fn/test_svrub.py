"""svrub is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrub import rubinstein_sp


def test_svrub_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rubinstein_sp(data=None)
