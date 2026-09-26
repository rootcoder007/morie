"""maphz is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.maphz import maphz


def test_maphz_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        maphz()
