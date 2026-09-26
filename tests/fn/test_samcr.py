"""samcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.samcr import samcr


def test_samcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        samcr()
