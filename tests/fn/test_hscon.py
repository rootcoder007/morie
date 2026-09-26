"""hscon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hscon import hscon


def test_hscon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hscon()
