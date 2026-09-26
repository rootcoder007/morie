"""sored is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sored import sored


def test_sored_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sored()
