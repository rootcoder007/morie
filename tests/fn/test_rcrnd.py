"""rcrnd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcrnd import rcrnd


def test_rcrnd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcrnd()
