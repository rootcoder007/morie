"""nmwno is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmwno import wnominate


def test_nmwno_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wnominate(data=None)
