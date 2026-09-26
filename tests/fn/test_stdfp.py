"""stdfp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stdfp import stdfp


def test_stdfp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stdfp()
