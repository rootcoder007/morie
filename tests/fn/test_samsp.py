"""samsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.samsp import samsp


def test_samsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        samsp()
