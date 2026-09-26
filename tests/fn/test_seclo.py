"""seclo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclo import seclo


def test_seclo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclo()
