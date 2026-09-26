"""ctvert is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctvert import ctvert


def test_ctvert_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctvert()
