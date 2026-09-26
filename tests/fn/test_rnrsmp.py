"""rnrsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnrsmp import rnrsmp


def test_rnrsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnrsmp()
