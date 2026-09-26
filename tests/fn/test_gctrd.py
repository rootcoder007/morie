"""gctrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gctrd import gctrd


def test_gctrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gctrd()
