"""mttsp2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mttsp2 import mttsp2


def test_mttsp2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mttsp2()
