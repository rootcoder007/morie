"""stiag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stiag import stiag


def test_stiag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stiag()
