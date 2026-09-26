"""rnmask is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnmask import rnmask


def test_rnmask_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnmask()
