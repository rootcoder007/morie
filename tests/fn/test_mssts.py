"""mssts is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mssts import stress_s1


def test_mssts_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stress_s1(X=None)
