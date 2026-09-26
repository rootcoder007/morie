"""msst2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msst2 import stress_norm


def test_msst2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stress_norm(X=None)
