"""ppnch is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ppnch import ppnch


def test_ppnch_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ppnch()
