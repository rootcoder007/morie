"""swlowt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swlowt import swlowt


def test_swlowt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swlowt(W=None)
