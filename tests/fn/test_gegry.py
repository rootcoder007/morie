"""gegry is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gegry import gegry


def test_gegry_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gegry()
