"""ubedu is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubedu import ubedu


def test_ubedu_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubedu()
