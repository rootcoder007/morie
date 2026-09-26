"""mttwr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mttwr import mttwr


def test_mttwr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mttwr()
