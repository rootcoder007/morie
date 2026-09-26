"""mhsim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mhsim import mhsim


def test_mhsim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mhsim()
