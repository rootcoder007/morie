"""csrsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csrsp import csrsp


def test_csrsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csrsp()
