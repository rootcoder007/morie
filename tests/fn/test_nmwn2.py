"""nmwn2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmwn2 import wnominate_2d


def test_nmwn2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wnominate_2d(data=None)
