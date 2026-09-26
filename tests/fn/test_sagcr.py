"""sagcr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sagcr import sagcr


def test_sagcr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sagcr()
