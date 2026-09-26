"""ctclip is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ctclip import ctclip


def test_ctclip_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ctclip()
