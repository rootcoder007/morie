"""gciod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gciod import gciod


def test_gciod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gciod()
