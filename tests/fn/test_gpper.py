"""gpper is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gpper import gpper


def test_gpper_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gpper()
