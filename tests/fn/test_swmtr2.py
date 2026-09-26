"""swmtr2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swmtr2 import swmtr2


def test_swmtr2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swmtr2(W=None)
