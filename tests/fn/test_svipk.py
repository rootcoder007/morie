"""svipk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svipk import ideal_point_kernel


def test_svipk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_kernel(data=None)
