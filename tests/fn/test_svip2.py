"""svip2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svip2 import ideal_point_2d


def test_svip2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_2d(data=None)
