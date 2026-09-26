"""svipn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svipn import ideal_point_normal


def test_svipn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ideal_point_normal(data=None)
