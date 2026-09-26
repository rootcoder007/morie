"""msst1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msst1 import stress_raw


def test_msst1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stress_raw(X=None)
