"""ubcmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubcmp import ubcmp


def test_ubcmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubcmp()
