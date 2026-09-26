"""seclr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclr import seclr


def test_seclr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclr()
