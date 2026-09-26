"""sebcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sebcr import sebcr


def test_sebcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sebcr()
