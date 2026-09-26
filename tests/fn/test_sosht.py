"""sosht is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sosht import sosht


def test_sosht_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sosht()
