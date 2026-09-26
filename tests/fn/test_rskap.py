"""rskap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rskap import rskap


def test_rskap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rskap()
