"""trelc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trelc import trelc


def test_trelc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trelc()
