"""stpor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stpor import stpor


def test_stpor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stpor()
