"""svrmv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svrmv import rm_intensity


def test_svrmv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rm_intensity(data=None)
