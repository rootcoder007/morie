"""jordCD is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.jordCD import jordan_canonical


def test_jordCD_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        jordan_canonical(A=None)
