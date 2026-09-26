"""gadrnp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gadrnp import gadrnp


def test_gadrnp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gadrnp()
