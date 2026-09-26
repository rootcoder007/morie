"""cdreal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdreal import cdreal


def test_cdreal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdreal()
