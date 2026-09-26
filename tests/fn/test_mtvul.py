"""mtvul is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtvul import mtvul


def test_mtvul_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtvul()
