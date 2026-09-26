"""gwrr2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gwrr2 import gwrr2


def test_gwrr2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwrr2(y=None, y_hat=None)
