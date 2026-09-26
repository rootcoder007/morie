"""swdense is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swdense import swdense


def test_swdense_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swdense(W=None)
