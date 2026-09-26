"""zecrs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zecrs import carstairs_index


def test_zecrs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        carstairs_index(data=None)
