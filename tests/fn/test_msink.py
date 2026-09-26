"""msink is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msink import indscal_3way


def test_msink_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        indscal_3way(data=None)
