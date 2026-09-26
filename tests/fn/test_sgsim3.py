"""sgsim3 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgsim3 import sgsim3


def test_sgsim3_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgsim3()
