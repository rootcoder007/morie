"""ovrsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ovrsmp import ovrsmp


def test_ovrsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ovrsmp()
