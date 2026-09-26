"""dk3sm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dk3sm import dk3sm


def test_dk3sm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dk3sm()
