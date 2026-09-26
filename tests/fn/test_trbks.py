"""trbks is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trbks import trbks


def test_trbks_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trbks()
