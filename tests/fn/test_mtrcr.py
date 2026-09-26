"""mtrcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtrcr import mtrcr


def test_mtrcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtrcr()
