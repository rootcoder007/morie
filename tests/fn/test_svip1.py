"""svip1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svip1 import ideal_point_1d


def test_svip1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_1d(data=None)
