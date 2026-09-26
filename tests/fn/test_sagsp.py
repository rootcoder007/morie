"""sagsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sagsp import sagsp


def test_sagsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sagsp()
