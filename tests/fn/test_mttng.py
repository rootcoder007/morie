"""mttng is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mttng import mttng


def test_mttng_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mttng()
