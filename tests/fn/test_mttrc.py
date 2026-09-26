"""mttrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mttrc import mttrc


def test_mttrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mttrc()
