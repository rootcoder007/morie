"""isngm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isngm import isngm


def test_isngm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isngm()
