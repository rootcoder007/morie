"""elimp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.elimp import elimp


def test_elimp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        elimp()
