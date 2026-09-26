"""rfwht is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rfwht import rfwht


def test_rfwht_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rfwht()
