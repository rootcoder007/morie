"""vdang is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vdang import vdang


def test_vdang_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vdang()
