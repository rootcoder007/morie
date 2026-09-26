"""simdk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.simdk import simdk


def test_simdk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        simdk()
